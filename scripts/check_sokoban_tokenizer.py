"""Exercise the real verl/Qwen tokenizer with scripted actions, without loading weights.

Run with verl/.venv/bin/python after installing ProbeGRPO in that environment.
This is an integration check, not model-generated success evidence.
"""

import asyncio
import json
import tempfile
from types import SimpleNamespace


async def main():
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap

    from probegrpo.envs.sokoban import SokobanEnv
    from probegrpo.integration.sokoban_agent_loop import SokobanAgentLoop

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B", local_files_only=True)

    class ScriptedServer:
        async def generate(self, **kwargs):
            ids = tokenizer.encode("up", add_special_tokens=False) + [tokenizer.eos_token_id]
            assert len(ids) <= kwargs["sampling_params"]["max_tokens"]
            return SimpleNamespace(token_ids=ids, log_probs=[-0.1] * len(ids), extra_fields={})

    with tempfile.TemporaryDirectory(prefix="probegrpo-tokenizer-") as output_dir:
        config = OmegaConf.create(
            {
                "actor_rollout_ref": {"rollout": {"prompt_length": 1024, "response_length": 2048}},
                "trainer": {"rollout_data_dir": output_dir},
            }
        )
        agent = SokobanAgentLoop(
            trainer_config=DictConfigWrap(config),
            server_manager=ScriptedServer(),
            tokenizer=tokenizer,
            processor=None,
            dataset_cls=None,
            data_config=DictConfigWrap(
                OmegaConf.create(
                    {
                        "apply_chat_template_kwargs": {"enable_thinking": False},
                    }
                )
            ),
            hf_model_type="qwen3_5",
        )
        output = await agent.run(
            {},
            uid="tokenizer-check",
            session_id=0,
            global_steps=0,
            extra_info={"task_id": "push-up", "env_seed": 17},
        )
        episode = json.loads(output.extra_fields["probegrpo_episode_json"])
        turns = episode["turns"]
        assert len(turns) == 2 and output.reward_score == 1.0
        assert len(output.response_mask) == len(output.response_ids)
        assert len(output.response_logprobs) == len(output.response_ids)
        owned = {i for turn in turns for i in turn["token_indices"]}
        assert owned == {i for i, flag in enumerate(output.response_mask) if flag}
        assert 0 in output.response_mask and output.response_mask[-1] == 1
        for turn in turns:
            state = SokobanEnv().replay("push-up", 17, turn["action_prefix"])
            assert state.state_hash == turn["state_hash"]
        # Validate the actual verl conversion path as well as the Pydantic schema.
        data = output.as_dict()
        assert float(data["rm_scores"].sum()) == 1.0
        print(
            f"Real Qwen CT tokenizer check passed: turns=2, "
            f"response_tokens={len(output.response_ids)}, assistant_tokens={len(owned)}"
        )
        print("Actions and logprobs were scripted; no model rollout was executed.")


if __name__ == "__main__":
    asyncio.run(main())
