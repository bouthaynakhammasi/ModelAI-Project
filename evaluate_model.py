import tensorflow as tf
import numpy as np
from PIL import Image
import os

LABELS = ["Atopic dermatitis","Cradle cap","Diaper rash","Drool rash","HFMD","Hemangioma","Impetigo","Miliaria","Pityriasis","Rubeola","Varicella"]
TEST_DIR = "./test_image"
MODEL_PATH = "./converted_model.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
print("Input dtype:", inp['dtype'])

total_correct = 0
total_images = 0

for disease in sorted(os.listdir(TEST_DIR)):
    disease_path = os.path.join(TEST_DIR, disease)
    if not os.path.isdir(disease_path): continue
    correct = 0
    total = 0
    for img_file in os.listdir(disease_path):
        if not img_file.lower().endswith(('.jpg','.jpeg','.png')): continue
        try:
            image = Image.open(os.path.join(disease_path, img_file)).convert("RGB").resize((224,224))
            arr = np.expand_dims(np.array(image, dtype=np.uint8), axis=0)
            interpreter.set_tensor(inp['index'], arr)
            interpreter.invoke()
            preds_raw = interpreter.get_tensor(out['index'])[0].astype(np.float32)
            if out['dtype'] == np.uint8:
                scale, zp = out['quantization']
                preds_raw = (preds_raw - zp) * scale
            exp_p = np.exp(preds_raw - np.max(preds_raw))
            preds = exp_p / exp_p.sum()
            predicted = LABELS[np.argmax(preds)]
            if predicted == disease: correct += 1
            total += 1
        except Exception as e:
            print("Error:", img_file[:30], e)
    if total > 0:
        acc = correct/total*100
        total_correct += correct
        total_images += total
        print(f"{'OK' if acc>=70 else 'XX'} {disease}: {correct}/{total} = {acc:.1f}%")

print(f"\nACCURACY GLOBALE: {total_correct}/{total_images} = {total_correct/total_images*100:.1f}%")