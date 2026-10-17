"""CPU numerical regressions, not a long-run GPU memory acceptance claim."""

import ast
import hashlib
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from probegrpo.integration.entropy import chunked_dense_entropy

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("dense_entropy_installer", ROOT / "scripts/install_verl_dense_entropy.py")
INSTALLER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSTALLER)


@unittest.skipIf(torch is None, "torch is not installed")
class DenseEntropyTest(unittest.TestCase):
    @staticmethod
    def entropy(logits):
        pd = torch.nn.functional.softmax(logits, dim=-1)
        return torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)

    def test_values_shape_dtype_and_gradients(self):
        torch.manual_seed(17)
        for dtype in (torch.float64, torch.float32, torch.bfloat16):
            for shape in ((17, 29), (2, 17, 29), (2, 3, 17, 29)):
                with self.subTest(dtype=dtype, shape=shape):
                    logits = torch.randn(shape, dtype=dtype, requires_grad=True)
                    before = logits.detach().clone()
                    dense = self.entropy(logits)
                    chunks = chunked_dense_entropy(logits, entropy_fn=self.entropy, chunk_size=5)
                    self.assertEqual(chunks.shape, logits.shape[:-1])
                    self.assertEqual(chunks.dtype, dtype)
                    self.assertEqual(chunks.device, logits.device)
                    tolerance = {torch.bfloat16: 0.01, torch.float32: 1e-6, torch.float64: 1e-12}[dtype]
                    torch.testing.assert_close(chunks, dense, rtol=tolerance, atol=tolerance)
                    dense_grad = torch.autograd.grad(dense.sum(), logits, retain_graph=True)[0]
                    chunk_grad = torch.autograd.grad(chunks.sum(), logits)[0]
                    torch.testing.assert_close(chunk_grad, dense_grad, rtol=tolerance, atol=tolerance)
                    self.assertTrue(torch.equal(before, logits.detach()))

    def test_noncontiguous_inputs_share_storage_and_bound_token_rows(self):
        for logits in (torch.randn(2, 257, 33).transpose(0, 1),
                       torch.randn(2, 257, 66)[..., ::2]):
            calls = []
            def measured(part):
                self.assertEqual(part.ndim, 2)
                self.assertLessEqual(part.shape[0], 7)
                self.assertEqual(part.untyped_storage().data_ptr(), logits.untyped_storage().data_ptr())
                calls.append(part.shape[0])
                return self.entropy(part)
            result = chunked_dense_entropy(logits, entropy_fn=measured, chunk_size=7)
            torch.testing.assert_close(result, self.entropy(logits))
            self.assertEqual(sum(calls), logits.numel() // logits.shape[-1])

    def test_checkpoint_forward_and_backward(self):
        from functools import partial
        from torch.utils.checkpoint import checkpoint
        fn = partial(chunked_dense_entropy, entropy_fn=self.entropy, chunk_size=5)
        for reentrant in (True, False):
            logits = torch.randn(2, 17, 29, requires_grad=True)
            actual = checkpoint(fn, logits, use_reentrant=reentrant)
            expected = self.entropy(logits)
            torch.testing.assert_close(actual, expected)
            actual_grad = torch.autograd.grad(actual.sum(), logits)[0] if not reentrant else None
            if reentrant:
                actual.sum().backward()
                actual_grad = logits.grad.clone()
            expected_grad = torch.autograd.grad(expected.sum(), logits)[0]
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_empty_tokens_and_invalid_arguments(self):
        logits = torch.empty(2, 0, 3, requires_grad=True)
        self.assertEqual(tuple(chunked_dense_entropy(logits, entropy_fn=self.entropy, chunk_size=3).shape), (2, 0))
        for size in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                chunked_dense_entropy(logits, entropy_fn=self.entropy, chunk_size=size)
        with self.assertRaises(ValueError):
            chunked_dense_entropy(torch.empty(3), entropy_fn=self.entropy, chunk_size=1)

    @unittest.skipUnless(os.environ.get("PROBEGRPO_TEST_VERL_DIR"), "optional pinned verl checkout not supplied")
    def test_actual_patched_engine_dispatch_with_and_without_checkpoint(self):
        import torch.utils.checkpoint
        path = Path(os.environ["PROBEGRPO_TEST_VERL_DIR"]) / INSTALLER.TARGET
        self.assertEqual(INSTALLER.sha256(path), INSTALLER.PATCHED_SHA256)
        tree = ast.parse(path.read_text())
        candidates = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                      and isinstance(node.test, ast.Name) and node.test.id == "calculate_entropy"
                      and "from probegrpo.integration.entropy import" in ast.unparse(node)]
        self.assertEqual(len(candidates), 1)
        code = compile(ast.fix_missing_locations(ast.Module(body=candidates, type_ignores=[])), str(path), "exec")
        for chunked in (False, True):
            for checkpointing in (False, True):
                logits = torch.randn(2, 17, 29, requires_grad=True)
                rows = []
                def measured(part):
                    rows.append(tuple(part.shape))
                    return self.entropy(part)
                config = SimpleNamespace(entropy_from_logits_with_chunking=chunked,
                                         entropy_from_logits_chunk_size=5,
                                         entropy_checkpointing=checkpointing)
                namespace = {"self": SimpleNamespace(engine_config=config), "logits": logits,
                             "calculate_entropy": True, "torch": torch,
                             "verl_F": SimpleNamespace(entropy_from_logits=measured)}
                exec(code, namespace)
                actual = namespace["entropy"]
                actual.sum().backward()
                expected = self.entropy(logits)
                expected_grad = torch.autograd.grad(expected.sum(), logits)[0]
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(logits.grad, expected_grad)
                if chunked:
                    self.assertTrue(all(len(shape) == 2 and shape[0] <= 5 for shape in rows))
                else:
                    self.assertTrue(all(shape == (2, 17, 29) for shape in rows))


class DenseEntropyInstallerTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.target = self.root / INSTALLER.TARGET
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(b"upstream fixture")
        settings = patch.multiple(INSTALLER,
                                  UPSTREAM_SHA256=hashlib.sha256(b"upstream fixture").hexdigest(),
                                  PATCHED_SHA256=hashlib.sha256(b"patched fixture").hexdigest())
        settings.start()
        self.addCleanup(settings.stop)
        self.git = patch.object(INSTALLER.subprocess, "check_output", side_effect=self.git_output)
        self.git.start()
        self.addCleanup(self.git.stop)

    def git_output(self, command, **kwargs):
        return str(self.root) if command[-1] == "--show-toplevel" else INSTALLER.VERL_COMMIT

    def test_install_is_exact_and_idempotent(self):
        def apply(command, **kwargs):
            if "--check" not in command:
                self.target.write_bytes(b"patched fixture")
        with patch.object(INSTALLER.subprocess, "run", side_effect=apply) as execute:
            self.assertIn("Installed", INSTALLER.install(self.root))
            self.assertEqual(execute.call_count, 2)
            self.assertIn("already installed", INSTALLER.install(self.root))
            self.assertEqual(execute.call_count, 2)
            INSTALLER.install(self.root, check_only=True)

    def test_check_only_and_unknown_edits_do_not_write(self):
        with patch.object(INSTALLER.subprocess, "run") as execute:
            with self.assertRaisesRegex(RuntimeError, "not installed"):
                INSTALLER.install(self.root, check_only=True)
            self.target.write_bytes(b"user edits")
            with self.assertRaisesRegex(RuntimeError, "preserving"):
                INSTALLER.install(self.root)
            self.assertEqual(self.target.read_bytes(), b"user edits")
            execute.assert_not_called()

    def test_wrong_commit_and_modified_patch_are_refused(self):
        with patch.object(INSTALLER.subprocess, "run") as execute:
            with patch.object(INSTALLER.subprocess, "check_output", side_effect=[str(self.root), "wrong"]):
                with self.assertRaisesRegex(RuntimeError, "pinned standalone"):
                    INSTALLER.install(self.root)
            with patch.object(INSTALLER, "PATCH_SHA256", "wrong"):
                with self.assertRaisesRegex(RuntimeError, "patch checksum"):
                    INSTALLER.install(self.root)
            execute.assert_not_called()

    def test_exact_patch_file_and_dense_checkpoint_dispatch(self):
        self.assertEqual(INSTALLER.sha256(INSTALLER.PATCH), INSTALLER.PATCH_SHA256)
        additions = "\n".join(line[1:] for line in INSTALLER.PATCH.read_text().splitlines()
                              if line.startswith("+") and not line.startswith("+++"))
        self.assertIn("chunked_dense_entropy", additions)
        self.assertIn("self.engine_config.entropy_from_logits_chunk_size", additions)
        self.assertIn("entropy = entropy_fn(logits)", additions)
        self.assertIn("checkpoint(entropy_fn, logits)", additions)


if __name__ == "__main__":
    unittest.main()
