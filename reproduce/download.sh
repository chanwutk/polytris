#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 || $1 != /* ]]; then
    echo 'Usage: bash reproduce/download.sh /absolute/data-parent [all|b3d|otif]' >&2
    exit 2
fi
case "${2:-all}" in
    all|b3d|otif) selection=${2:-all} ;;
    *) echo 'Choose all, b3d, or otif.' >&2; exit 2 ;;
esac
if ! command -v hf >/dev/null; then
    echo 'Install the Hugging Face CLI (hf) on the Docker host.' >&2
    exit 1
fi

datasets="$1/polyis-data/datasets"
cache="$1/polyis-cache"
scratch=$(mktemp -d)
staged=''
cleanup() {
    rm -rf -- "$scratch"
    if [[ -n $staged ]]; then rm -f -- "$staged"; fi
}
trap cleanup EXIT

stage_tree() {
    local source=$1 destination=$2 input target parent
    find "$source" -type f -print0 > "$scratch/files"
    while IFS= read -r -d '' input; do
        target="$destination/${input#"$source/"}"
        if [[ ! -e $target && ! -L $target ]]; then
            parent=$(dirname -- "$target")
            mkdir -p -- "$parent"
            staged=$(mktemp "$parent/.tetris-download.XXXXXX")
            cp -p -- "$input" "$staged"
            mv -n -- "$staged" "$target"
            rm -f -- "$staged"
            staged=''
        fi
        if [[ ! -f $target || -L $target ]] || ! cmp -s -- "$input" "$target"; then
            printf 'Existing file differs from the release: %s\n' "$target" >&2
            return 1
        fi
    done < "$scratch/files"
}

download_bundle() {
    hf download "$1" --repo-type dataset --revision "$2" --local-dir "$scratch/bundle"
    stage_tree "$scratch/bundle/datasets" "$datasets"
    stage_tree "$scratch/bundle/groundtruth" "$cache"
    rm -rf -- "$scratch/bundle"
}

if [[ $selection != otif ]]; then
    download_bundle tetris-db/b3d-processed 8b169c08266c8c78515112735051067f2a4abd4c
fi
if [[ $selection != b3d ]]; then
    download_bundle tetris-db/otif-processed 98d1809eb4d367032e878f53bdd584103776e34a
    for dataset in caldot1 caldot2 ams; do
        alias_path="$datasets/$dataset-y05"
        if [[ -L $alias_path && $(readlink -- "$alias_path") == "$dataset" ]]; then
            continue
        fi
        if [[ -e $alias_path || -L $alias_path ]]; then
            printf 'Cannot create alias over an existing path: %s\n' "$alias_path" >&2
            exit 1
        fi
        ln -s -- "$dataset" "$alias_path"
    done
fi
