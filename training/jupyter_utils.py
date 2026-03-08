import io
import numpy as np
from PIL import Image
import ipywidgets as widgets
from IPython.display import display, clear_output
import tensorflow as tf

def load_and_preprocess_image(contents):
    img = Image.open(io.BytesIO(contents))
    img = img.resize((256, 256))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def run_prediction_ui(model, class_names):
    upload_btn = widgets.FileUpload(accept='image/*', multiple=false)
    output = widgets.Output()

    def on_upload_change(change):
        with output:
            clear_output()
            if not upload_btn.value:
                return
            
            # handle both old and new ipywidgets versions
            file_info = list(upload_btn.value.values())[0] if isinstance(upload_btn.value, dict) else upload_btn.value[0]
            content = file_info['content']
            
            # Display image
            display(Image.open(io.BytesIO(content)))
            
            # Predict
            img_array = load_and_preprocess_image(content)
            predictions = model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions[0])]
            confidence = np.max(predictions[0])
            
            print(f"Prediction: {predicted_class}")
            print(f"Confidence: {confidence:.2%}")

    upload_btn.observe(on_upload_change, names='value')
    display(upload_btn, output)
