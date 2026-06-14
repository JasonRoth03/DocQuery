## Chunking Strategy
1. Recursively Split:  Try separators in order (\n\n → \n → ". " → " " → char), if a piece stiil exceeds the set chunk size tokens after splittin on the seperator then it recurses to a finer seperator and finally falls back to raw token-boundary slicing as a last resort
2. Merge with overlap: packs the atomic pieces into full size chunks. When adding the next piecve would overflow chunk size, it flushes the current chunk and seeds the next one with the last overlap tokens from the flushed chunk.

