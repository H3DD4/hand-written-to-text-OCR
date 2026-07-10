import asyncio
from ocr import extract_text
from llm import analyse_with_models_parallel

pic_path = r"c:\Users\MSI\OneDrive\Bureau\talan_rag\pic\e61abc66-061f-4860-ae58-e3211c0f7bc9.jpeg"

async def main():
    print(f"Reading {pic_path}...")

    with open(pic_path, 'rb') as f:
        # OCR is synchronous, wrapped in to_thread if in async context,
        # but here we can just call it synchronously since it's the only task running
        text_out = extract_text(f.read(), pic_path)

    print("\n" + "="*40)
    print("=== 1. OCR TEXT EXTRACTED ===")
    print("="*40)
    print(text_out)

    print("\n" + "="*40)
    print("=== 2. PARALLEL LLM ANALYSIS ===")
    print("="*40)
    try:
        res = await analyse_with_models_parallel(text_out)
        import json
        print(json.dumps(res, indent=2, ensure_ascii=False))
    except Exception as e:
        print("LLM Error:", e)

if __name__ == "__main__":
    asyncio.run(main())
