test_img = "/content/drive/MyDrive/c.png"
label, conf, probs = infer_image(test_img, enhanced_model, class_names)
nsfw_score = compute_enhanced_nsfw_score(Image.open(test_img).convert("RGB"))

if nsfw_score > 0.7:
    print("⚠ Classified as Adult Content (NSFW)")
else:
    print(f"Classified as {label} with confidence {conf * 100:.2f}%")

print("All probs:", f"{probs*100:.1f}%")
