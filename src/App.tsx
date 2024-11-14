import {
  env,
  AutoModel,
  AutoProcessor,
  RawImage,
} from "@huggingface/transformers";
import { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { useDropzone } from "react-dropzone";
import { Loader2, Download } from "lucide-react"; // Add to imports
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue 
} from "@/components/ui/select";

env.backends.onnx.wasm.proxy = false;

let model, processor;

const initializeModel = async () => {
  if (!model || !processor) {
    model = await AutoModel.from_pretrained('briaai/RMBG-1.4', {
      config: { model_type: 'custom' },
    });
    processor = await AutoProcessor.from_pretrained('briaai/RMBG-1.4', {
        config: {
            do_normalize: true,
            do_pad: false,
            do_rescale: true,
            do_resize: true,
            image_mean: [0.5, 0.5, 0.5],
            feature_extractor_type: "ImageFeatureExtractor",
            image_std: [1, 1, 1],
            resample: 2,
            rescale_factor: 0.00392156862745098,
            size: { width: 1024, height: 1024 },
        }
    });
  }
};

async function processImage(image) {
  const img = await RawImage.fromURL(URL.createObjectURL(image));
  
  const { pixel_values } = await processor(img);
  
  const { output } = await model({ input: pixel_values });

  const maskData = (
    await RawImage.fromTensor(output[0].mul(255).to("uint8")).resize(
      img.width,
      img.height
    )
  ).data;

  const canvas = document.createElement("canvas");
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Could not get 2D context");

  ctx.drawImage(img.toCanvas(), 0, 0);

  const pixelData = ctx.getImageData(0, 0, img.width, img.height);
  for (let i = 0; i < maskData.length; ++i) {
    pixelData.data[4 * i + 3] = maskData[i];
  }
  ctx.putImageData(pixelData, 0, 0);

  const blob = await new Promise((resolve, reject) => 
    canvas.toBlob(blob => (blob ? resolve(blob) : reject()), "image/png")
  );
  return new File([blob], `${image.name.split(".")[0]}-bg-removed.png`, { type: "image/png" });
}

const countries = {
  US: "United States",
  GB: "United Kingdom",
  FR: "France",
  DE: "Germany",
  IT: "Italy",
  ES: "Spain",
  JP: "Japan",
  CN: "China",
  // Add more countries as needed
};

export default function App() {
  const [originalImage, setOriginalImage] = useState<string | null>(null);
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [selectedCountry, setSelectedCountry] = useState<string | null>(null);

  useEffect(() => {
    initializeModel().catch(err => setError("Failed to initialize model: " + err.message));
  }, []);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setIsLoading(true);
    setProgress(0);

    setOriginalImage(URL.createObjectURL(file));

    try {
      setProgress(30);
      const img = new Image();
      img.src = URL.createObjectURL(file);
      await img.decode();

      setProgress(60);
      const processedFile = await processImage(file);
      setProcessedImage(URL.createObjectURL(processedFile));

      setProgress(100);
      setIsLoading(false);
    } catch (error) {
      console.error(error);
      setIsLoading(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.webp']
    },
    multiple: false
  });

  // Add function to combine images
  const saveCompositeImage = async () => {
    if (!processedImage || !selectedCountry) return;

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    // Create new images
    const bgImage = new Image();
    const fgImage = new Image();
    
    // Load flag background
    bgImage.src = `https://cdn.jsdelivr.net/npm/svg-country-flags@1.2.10/svg/${selectedCountry.toLowerCase()}.svg`;
    await bgImage.decode();
    
    // Load processed image
    fgImage.src = processedImage;
    await fgImage.decode();
    
    // Set canvas size
    canvas.width = fgImage.width;
    canvas.height = fgImage.height;
    
    // Draw background flag
    ctx.drawImage(bgImage, 0, 0, canvas.width, canvas.height);
    // Draw transparent image on top
    ctx.drawImage(fgImage, 0, 0, canvas.width, canvas.height);
    
    // Create download link
    const link = document.createElement('a');
    link.download = 'combined-image.png';
    link.href = canvas.toDataURL('image/png');
    link.click();
  };

  return (
    <div className="min-h-screen w-screen bg-gray-50">
      <div className="container mx-auto p-4 md:p-8 space-y-6">
        <Card className="w-full p-6 md:p-8">
          <div
            {...getRootProps()}
            className={`
              w-full border-2 border-dashed rounded-lg p-8 md:p-12 text-center cursor-pointer
              transition-colors duration-200 flex flex-col items-center justify-center
              min-h-[200px]
              ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300'}
            `}
          >
            <input {...getInputProps()} />
            <div className="space-y-4">
              <div className="text-gray-600 text-lg">
                {isDragActive ? (
                  <p>Drop the image here...</p>
                ) : (
                  <p>Drag & drop an image here, or click to select</p>
                )}
              </div>
              <Button variant="outline" size="lg">
                Select Image
              </Button>
            </div>
          </div>

          {isLoading && (
            <div className="mt-6 w-full space-y-2">
              <Progress value={progress} className="w-full" />
              <div className="flex items-center justify-center gap-2 text-sm text-gray-600">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span>Processing image...</span>
              </div>
            </div>
          )}
        </Card>

        {processedImage && (
          <Card className="w-full p-6">
            <div className="space-y-4">
              <h3 className="font-medium text-lg">Select Background Flag</h3>
              <Select
                onValueChange={(value) => setSelectedCountry(value)}
                value={selectedCountry || undefined}
              >
                <SelectTrigger className="w-full md:w-[300px]">
                  <SelectValue placeholder="Choose a country flag" />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(countries).map(([code, name]) => (
                    <SelectItem key={code} value={code}>
                      <div className="flex items-center gap-2">
                        <img 
                          src={`https://cdn.jsdelivr.net/npm/svg-country-flags@1.2.10/svg/${code.toLowerCase()}.svg`}
                          alt={name}
                          className="w-6 h-4 object-cover"
                        />
                        {name}
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </Card>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 w-full">
          {originalImage && (
            <Card className="w-full">
              <div className="p-4 border-b">
                <h3 className="font-medium text-lg">Original Image</h3>
              </div>
              <div className="relative p-4">
                <img 
                  src={originalImage} 
                  alt="Original" 
                  className="w-full h-auto rounded-lg object-contain max-h-[600px]"
                />
              </div>
            </Card>
          )}

          {processedImage && (
            <Card className="w-full">
              <div className="p-4 border-b">
                <h3 className="font-medium text-lg">Background Removed</h3>
              </div>
              <div className="relative p-4">
                {selectedCountry && (
                  <img
                    src={`https://cdn.jsdelivr.net/npm/svg-country-flags@1.2.10/svg/${selectedCountry.toLowerCase()}.svg`}
                    alt="Flag background"
                    className="absolute inset-0 w-full h-full object-cover"
                  />
                )}
                <img 
                  src={processedImage} 
                  alt="Processed"
                  className="relative w-full h-auto rounded-lg object-contain max-h-[600px] z-10"
                />
              </div>
            </Card>
          )}
        </div>

        {processedImage && selectedCountry && (
          <Card className="w-full">
            <div className="p-4 border-b flex justify-between items-center">
              <h3 className="font-medium text-lg">Combined Result</h3>
              <Button
                onClick={saveCompositeImage}
                className="flex items-center gap-2"
              >
                <Download className="h-4 w-4" />
                Save Combined Image
              </Button>
            </div>
            <div className="relative p-4">
              {selectedCountry && (
                <img
                  src={`https://cdn.jsdelivr.net/npm/svg-country-flags@1.2.10/svg/${selectedCountry.toLowerCase()}.svg`}
                  alt="Flag background"
                  className="absolute inset-0 w-full h-full object-cover"
                />
              )}
              <img 
                src={processedImage} 
                alt="Processed"
                className="relative w-full h-auto rounded-lg object-contain max-h-[600px] z-10"
              />
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}
