from transformers import pipeline
import pandas as pd


def main():
    """Create captions with Blip model"""
    image_list = [f"./CelebAMask-HQ/CelebA-HQ-img/{i}.jpg" for i in range(30000)]
    new_captions = []
    captioner = pipeline("image-to-text", model="Salesforce/blip2-opt-2.7b", device="cuda:0", max_new_tokens=100)
    for i in range(300):
        caption = captioner(image_list[100 * i: 100 * (i + 1)])
        for c in caption:
            new_captions.append(c[0]["generated_text"].strip())
    df = pd.read_csv("captions.csv")
    df["blip2_caption"] = new_captions
    df.to_csv("new_captions.csv", index=False)


if __name__ == "__main__":
    main()
