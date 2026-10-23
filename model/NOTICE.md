# Source attribution

Model equations and checkpoint layout were independently expressed in a
small PyTorch implementation after studying:

- Hugging Face Transformers, `models/gemma4_unified`, commit
  `02d8fb9784e8f14a1251e4c992cd82a5762417c6`.
  Copyright 2026 the HuggingFace Team. Licensed under Apache License 2.0:
  <https://www.apache.org/licenses/LICENSE-2.0>.
- bitsandbytes NF4 codebook, version 0.50.2. MIT License; attribution below.

No upstream reference package is vendored or imported by the core model.
The optional accelerated quantized linear path imports bitsandbytes.

## bitsandbytes notice

MIT License

Copyright (c) Facebook, Inc. and its affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
