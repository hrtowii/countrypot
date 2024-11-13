import {
  env,
  AutoModel,
  AutoProcessor,
  RawImage,
} from "@huggingface/transformers";
import React, { useState, useEffect } from "react";

env.backends.onnx.wasm.proxy = false;

const model_id = "Xenova/modnet";
let model, processor;

// Initialize model and processor outside of component to avoid reloading
const initializeModel = async () => {
  if (!model || !processor) {
    model = await AutoModel.from_pretrained(model_id, { device: "webgpu" });
    processor = await AutoProcessor.from_pretrained(model_id);
  }
};

async function processImage(image) {
  const img = await RawImage.fromURL(URL.createObjectURL(image));
  
  // Pre-process image
  const { pixel_values } = await processor(img);
  
  // Predict alpha matte
  const { output } = await model({ input: pixel_values });

  const maskData = (
    await RawImage.fromTensor(output[0].mul(255).to("uint8")).resize(
      img.width,
      img.height
    )
  ).data;

  // Create new canvas and draw image
  const canvas = document.createElement("canvas");
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Could not get 2D context");

  ctx.drawImage(img.toCanvas(), 0, 0);

  // Apply alpha mask
  const pixelData = ctx.getImageData(0, 0, img.width, img.height);
  for (let i = 0; i < maskData.length; ++i) {
    pixelData.data[4 * i + 3] = maskData[i];
  }
  ctx.putImageData(pixelData, 0, 0);

  // Convert canvas to blob and create a new file
  const blob = await new Promise((resolve, reject) => 
    canvas.toBlob(blob => (blob ? resolve(blob) : reject()), "image/png")
  );
  return new File([blob], `${image.name.split(".")[0]}-bg-removed.png`, { type: "image/png" });
}

function App() {
  const [processedImage, setProcessedImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    initializeModel().catch(err => setError("Failed to initialize model: " + err.message));
  }, []);

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    setIsLoading(true);
    setError(null);

    try {
      const processedFile = await processImage(file);
      setProcessedImage(URL.createObjectURL(processedFile));
    } catch (err) {
      setError("Image processing failed: " + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div>
      <input type="file" onChange={handleImageUpload} accept="image/*" />
      {isLoading && <p>Loading...</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {processedImage && (
        <div>
          <p>Processed Image:</p>
          <img src={processedImage} alt="Processed result" />
        </div>
      )}
    </div>
  );
}

export default App;
