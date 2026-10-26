#!/bin/sh
set -eu
# Regenerate checked-in SPIR-V arrays after editing GameOfLifeNN/shaders.
# Normal CMake builds use the generated files and need no shader compiler.
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
for shader in agent.vert color.frag food.vert grid.vert grid.frag; do
    glslangValidator -V --target-env vulkan1.1 \
        "$ROOT/GameOfLifeNN/shaders/$shader" -o "/tmp/$shader.spv"
    spirv-val --target-env vulkan1.1 "/tmp/$shader.spv"
    python3 - "$ROOT" "$shader" <<'PY'
import pathlib, struct, sys
root, shader = sys.argv[1:]
data = pathlib.Path(f'/tmp/{shader}.spv').read_bytes()
words = struct.unpack(f'<{len(data)//4}I', data)
name = shader.replace('.', '_')
body = ', '.join(f'0x{word:08x}' for word in words)
pathlib.Path(f'{root}/GameOfLifeNN/src/Vulkan/{name}.spv.inc').write_text(
    f'inline constexpr unsigned int {name}_spv[] = {{\n{body}\n}};\n')
print(f'{shader}: {len(words)} SPIR-V words')
PY
done
