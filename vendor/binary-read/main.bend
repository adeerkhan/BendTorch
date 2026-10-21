# A local copy of bend-ml-binary-read@0.1.0.0 (bend-ml's binary-read/, MIT), for BendTorch's loading code only, with
# two additions: 64-bit byte offsets and sizes, and BF16 files. `read_f32`, `read_f32.c` and `read_f32.js` are the
# package's, unchanged; the additions are `size64`, `read_f32_at`, `read_bf16_at`, `read_f32_into` and `read_bf16_into`,
# in `read_at.c` and `read_at.js`.
# Whether they become a new version of the package is a separate decision; until then torch/load.bend imports this
# file by its path.
#
# Bend 2.0.37 has no 64-bit integer type (Base's numbers are U32, F32 and Nat), so a 64-bit byte offset or size
# crosses as two U32s, hi and lo, meaning hi * 2^32 + lo. An offset counted in 4-byte cells would have covered 16 GiB
# with one U32, but a tensor in a safetensors file starts at byte 8 + the header's length + its own offset, and the
# header has any length, so the offsets are bytes.
#
# The original package's header follows.
#
# bend-ml-binary-read: little-endian F32 numbers from a file, read straight into an Array<F32>.
#
#   import bend-ml-binary-read@0.1.0.0/main.bend as R
#
# File.read_at answers a List<U32> with one cell per byte, and building and walking that list is
# what loading model weights costs (NOTES.md, exp. 13; bendlang/bend#1573). This effect reads the
# bytes into the array's own cells instead, with no list in between. The idea and its shape come
# from bendlang/bend#1430 by @totaland, which proposed it for Base; Base keeps thin syscall
# wrappers, so it lives here as a library effect.
#
# It is foreign code: --verdict fails for any file that imports this one, used or not. Keep it out
# of files that hold laws, and import it only where weights are loaded.
import Base

# Reads n numbers (4 * n bytes) at byte `offset` of `file` into cells 0 .. n-1 of `a`, stopping at
# a's capacity and at the end of the file. Answers the file, the array and the number of bytes
# read, or the error; the file's position does not move. A short read leaves the cells past it as
# they were (a last partial number writes only its bytes).
def read_f32(file: File, offset: U32, n: U32, a: Array<F32>) ->
  IO(File & (Array<F32> & Result<&1, &1, U32 & String, U32>)):
  import "./read_f32.c"
  import "./read_f32.js"

# The size of `file` in bytes as (hi, lo): hi * 2^32 + lo. Base's File.size fails past 4 GiB.
def size64(file: File) ->
  IO(File & Result<&1, &1, U32 & String, U32 & U32>):
  import "./read_at.c"
  import "./read_at.js"

# As read_f32, at the byte offset hi * 2^32 + lo, and it answers the number of whole numbers read (not bytes), so a
# read of up to 2^32 - 1 numbers (16 GiB) answers its count in a U32. Reads n numbers into cells 0 .. n-1 of `a`,
# stopping at a's capacity and at the end of the file; the file's position does not move. A short read leaves the
# cells past it as they were, except that a last partial number writes only its bytes, as in read_f32.
def read_f32_at(file: File, hi: U32, lo: U32, n: U32, a: Array<F32>) ->
  IO(File & (Array<F32> & Result<&1, &1, U32 & String, U32>)):
  import "./read_at.c"
  import "./read_at.js"

# n little-endian BF16 numbers (2 * n bytes) at the byte offset hi * 2^32 + lo, each widened exactly into a cell of
# `a`: the F32 whose top 16 bits are the BF16's and whose low 16 bits are zero (what bf16.to(torch.float32) gives).
# Stops at a's capacity and at the end of the file, answers the number of whole numbers read, and leaves the cells
# past them as they were (a last odd byte is dropped). The file's position does not move.
def read_bf16_at(file: File, hi: U32, lo: U32, n: U32, a: Array<F32>) ->
  IO(File & (Array<F32> & Result<&1, &1, U32 & String, U32>)):
  import "./read_at.c"
  import "./read_at.js"

# As read_f32_at, writing cells at .. at + n - 1 of `a` (stopping at a's capacity): one tensor of a file into its place
# in a larger array, with no copy.
def read_f32_into(file: File, hi: U32, lo: U32, n: U32, a: Array<F32>, at: U32) ->
  IO(File & (Array<F32> & Result<&1, &1, U32 & String, U32>)):
  import "./read_at.c"
  import "./read_at.js"

# As read_bf16_at, writing cells at .. at + n - 1 of `a` (stopping at a's capacity).
def read_bf16_into(file: File, hi: U32, lo: U32, n: U32, a: Array<F32>, at: U32) ->
  IO(File & (Array<F32> & Result<&1, &1, U32 & String, U32>)):
  import "./read_at.c"
  import "./read_at.js"
