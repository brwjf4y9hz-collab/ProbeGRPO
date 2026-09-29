#/usr/bin/env bash
set -euo pipefail
PROJECTDIR="cd "dirname "BASHSOURCE0"/.." && pwd"
COMMIT="cat "PROJECTDIR/repro/verl.commit""
PATCH="PROJECTDIR/patches/verl-probegrpo.patch"
VERLDIR="VERLDIR-PROJECTDIR/../probegrpo-runtime/verl"
ARCHIVEURL="https//github.com/verl-project/verl/archive/COMMIT.tar.gz"

if  -e "VERLDIR"  then
  if   -d "VERLDIR/.git"  then
      echo "Refusing to modify existing non-git path VERLDIR" &2
          exit 2
            fi
              actual="git -C "VERLDIR" rev-parse HEAD"
                if  "actual" = "COMMIT"  then
                    echo "verl at VERLDIR is actual expected COMMIT. No files changed." &2
                        exit 2
                          fi
                            if git -C "VERLDIR" apply --reverse --check "PATCH" /dev/null 2&1 then
                                echo "ProbeGRPO patch is already applied in VERLDIR"
                                    exit 0
                                      fi
                                        git -C "VERLDIR" apply --check "PATCH"
                                          git -C "VERLDIR" apply "PATCH"
                                          else
                                            mkdir -p "dirname "VERLDIR""
                                              tmp="mktemp -d"
                                                trap 'rm -rf "tmp"' EXIT
                                                  curl --fail --location --retry 3 "ARCHIVEURL" -o "tmp/verl.tar.gz"
                                                    mkdir "VERLDIR"
                                                      tar -xzf "tmp/verl.tar.gz" --strip-components=1 -C "VERLDIR"
                                                        git -C "VERLDIR" init -q
                                                          git -C "VERLDIR" apply --check "PATCH"
                                                            git -C "VERLDIR" apply "PATCH"
                                                            fi
                                                            echo "Prepared verl COMMIT with ProbeGRPO patch at VERLDIR"
                                                            