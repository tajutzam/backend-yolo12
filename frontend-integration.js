/**
 * YOLOv12 Detection API - JavaScript/Frontend Integration Examples
 * 
 * Copy dan paste code ini ke aplikasi web Anda untuk mengintegrasikan API
 */

// ============================================================
// 1. Basic API Client Class
// ============================================================

class YOLOv12APIClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl.replace(/\/$/, '');
    }

    async healthCheck() {
        const response = await fetch(`${this.baseUrl}/health`);
        return await response.json();
    }

    async getAvailableModels() {
        const response = await fetch(`${this.baseUrl}/models`);
        return await response.json();
    }

    async predictFromFile(
        file,
        modelName = 'yolov12m.pt',
        confThreshold = 0.25,
        imgSize = 640
    ) {
        const formData = new FormData();
        formData.append('file', file);

        const params = new URLSearchParams({
            model_name: modelName,
            conf_threshold: confThreshold,
            img_size: imgSize
        });

        const response = await fetch(
            `${this.baseUrl}/predict?${params}`,
            { method: 'POST', body: formData }
        );

        if (!response.ok) {
            throw new Error(`API Error: ${response.statusText}`);
        }

        return await response.json();
    }

    async predictFromUrl(
        imageUrl,
        modelName = 'yolov12m.pt',
        confThreshold = 0.25,
        imgSize = 640
    ) {
        const params = new URLSearchParams({
            image_url: imageUrl,
            model_name: modelName,
            conf_threshold: confThreshold,
            img_size: imgSize
        });

        const response = await fetch(
            `${this.baseUrl}/predict-url?${params}`,
            { method: 'POST' }
        );

        if (!response.ok) {
            throw new Error(`API Error: ${response.statusText}`);
        }

        return await response.json();
    }

    async switchModel(modelName) {
        const params = new URLSearchParams({ model_name: modelName });
        const response = await fetch(
            `${this.baseUrl}/switch-model?${params}`,
            { method: 'POST' }
        );
        return await response.json();
    }
}

// ============================================================
// 2. HTML Example - Image Upload & Display Results
// ============================================================

/*
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLOv12 Detection</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana; background: #f5f5f5; }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        h1 { color: #333; margin-bottom: 30px; text-align: center; }
        
        .upload-section {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #333;
        }
        
        input[type="file"],
        input[type="url"],
        select,
        input[type="range"] {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }
        
        .range-value { display: inline-block; margin-left: 10px; color: #666; }
        
        button {
            background: #007bff;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
        }
        
        button:hover { background: #0056b3; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        
        .results-section {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-box {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 4px;
            text-align: center;
        }
        
        .stat-box .number { font-size: 32px; font-weight: bold; color: #007bff; }
        .stat-box .label { color: #666; margin-top: 5px; }
        
        .detections-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        
        .detections-table th {
            background: #f8f9fa;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }
        
        .detections-table td {
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }
        
        .confidence {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            color: white;
        }
        
        .confidence.high { background: #28a745; }
        .confidence.medium { background: #ffc107; }
        .confidence.low { background: #dc3545; }
        
        .loading { display: none; text-align: center; margin: 20px 0; }
        .loading.active { display: block; }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #007bff;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        
        .error { color: #dc3545; background: #f8d7da; padding: 12px; border-radius: 4px; margin: 10px 0; }
        .success { color: #155724; background: #d4edda; padding: 12px; border-radius: 4px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 YOLOv12 Object Detection</h1>
        
        <div class="upload-section">
            <div class="form-group">
                <label>Select Image</label>
                <input type="file" id="fileInput" accept="image/*">
            </div>
            
            <div class="form-group">
                <label>Or Enter Image URL</label>
                <input type="url" id="urlInput" placeholder="https://example.com/image.jpg">
            </div>
            
            <div class="form-group">
                <label>Model</label>
                <select id="modelSelect">
                    <option value="yolov12n.pt">YOLOv12 Nano (Fast)</option>
                    <option value="yolov12s.pt">YOLOv12 Small</option>
                    <option value="yolov12m.pt" selected>YOLOv12 Medium (Default)</option>
                    <option value="yolov12l.pt">YOLOv12 Large</option>
                    <option value="yolov12x.pt">YOLOv12 XLarge (Accurate)</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>Confidence Threshold: <span class="range-value" id="confValue">0.25</span></label>
                <input type="range" id="confSlider" min="0" max="1" step="0.05" value="0.25">
            </div>
            
            <div class="form-group">
                <label>Image Size: <span class="range-value" id="sizeValue">640</span></label>
                <input type="range" id="sizeSlider" min="320" max="1280" step="32" value="640">
            </div>
            
            <button onclick="predict()">🚀 Detect Objects</button>
            <div class="loading" id="loading"><div class="spinner"></div></div>
            <div id="message"></div>
        </div>
        
        <div class="results-section" id="resultsSection" style="display: none;">
            <h2>Results</h2>
            
            <div class="stats">
                <div class="stat-box">
                    <div class="number" id="detectionCount">0</div>
                    <div class="label">Objects Detected</div>
                </div>
                <div class="stat-box">
                    <div class="number" id="modelUsed">-</div>
                    <div class="label">Model</div>
                </div>
                <div class="stat-box">
                    <div class="number" id="avgConfidence">0%</div>
                    <div class="label">Avg Confidence</div>
                </div>
            </div>
            
            <table class="detections-table" id="detectionsTable">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Class</th>
                        <th>Confidence</th>
                        <th>BBox Coordinates</th>
                        <th>Size</th>
                    </tr>
                </thead>
                <tbody id="detectionsBody">
                </tbody>
            </table>
        </div>
    </div>
    
    <script src="yolov12-client.js"></script>
    <script>
        const client = new YOLOv12APIClient();
        
        // Update range value display
        document.getElementById('confSlider').addEventListener('input', (e) => {
            document.getElementById('confValue').textContent = e.target.value;
        });
        
        document.getElementById('sizeSlider').addEventListener('input', (e) => {
            document.getElementById('sizeValue').textContent = e.target.value;
        });
        
        async function predict() {
            const fileInput = document.getElementById('fileInput');
            const urlInput = document.getElementById('urlInput');
            const modelSelect = document.getElementById('modelSelect');
            const confSlider = document.getElementById('confSlider');
            const sizeSlider = document.getElementById('sizeSlider');
            const messageDiv = document.getElementById('message');
            const loadingDiv = document.getElementById('loading');
            const resultsSection = document.getElementById('resultsSection');
            
            messageDiv.innerHTML = '';
            
            try {
                loadingDiv.classList.add('active');
                
                let result;
                
                if (fileInput.files.length > 0) {
                    result = await client.predictFromFile(
                        fileInput.files[0],
                        modelSelect.value,
                        parseFloat(confSlider.value),
                        parseInt(sizeSlider.value)
                    );
                } else if (urlInput.value) {
                    result = await client.predictFromUrl(
                        urlInput.value,
                        modelSelect.value,
                        parseFloat(confSlider.value),
                        parseInt(sizeSlider.value)
                    );
                } else {
                    throw new Error('Please upload a file or enter an image URL');
                }
                
                displayResults(result);
                resultsSection.style.display = 'block';
                messageDiv.innerHTML = '<div class="success">✓ Detection completed successfully!</div>';
                
            } catch (error) {
                messageDiv.innerHTML = `<div class="error">✗ Error: ${error.message}</div>`;
                resultsSection.style.display = 'none';
            } finally {
                loadingDiv.classList.remove('active');
            }
        }
        
        function displayResults(result) {
            document.getElementById('detectionCount').textContent = result.detection_count;
            document.getElementById('modelUsed').textContent = result.model;
            
            const tbody = document.getElementById('detectionsBody');
            tbody.innerHTML = '';
            
            let totalConfidence = 0;
            result.detections.forEach((det, idx) => {
                const confPercent = (det.confidence * 100).toFixed(1);
                const confClass = det.confidence > 0.75 ? 'high' : 
                                 det.confidence > 0.5 ? 'medium' : 'low';
                
                totalConfidence += det.confidence;
                
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${idx + 1}</td>
                    <td><strong>${det.class_name}</strong></td>
                    <td><span class="confidence ${confClass}">${confPercent}%</span></td>
                    <td>(${det.bbox.x1.toFixed(0)}, ${det.bbox.y1.toFixed(0)}) → (${det.bbox.x2.toFixed(0)}, ${det.bbox.y2.toFixed(0)})</td>
                    <td>${det.width.toFixed(0)}x${det.height.toFixed(0)}</td>
                `;
                tbody.appendChild(row);
            });
            
            const avgConf = result.detection_count > 0 
                ? ((totalConfidence / result.detection_count) * 100).toFixed(1)
                : 0;
            document.getElementById('avgConfidence').textContent = avgConf + '%';
        }
    </script>
</body>
</html>
*/

// ============================================================
// 3. React Component Example
// ============================================================

/*
import React, { useState, useEffect } from 'react';

const YOLOv12Detector = () => {
    const [client] = useState(() => new YOLOv12APIClient());
    const [file, setFile] = useState(null);
    const [url, setUrl] = useState('');
    const [model, setModel] = useState('yolov12m.pt');
    const [confidence, setConfidence] = useState(0.25);
    const [imageSize, setImageSize] = useState(640);
    const [results, setResults] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    
    const predict = async () => {
        setError('');
        setLoading(true);
        
        try {
            let result;
            
            if (file) {
                result = await client.predictFromFile(file, model, confidence, imageSize);
            } else if (url) {
                result = await client.predictFromUrl(url, model, confidence, imageSize);
            } else {
                setError('Please provide an image file or URL');
                setLoading(false);
                return;
            }
            
            setResults(result);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };
    
    return (
        <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '20px' }}>
            <h1>🎯 YOLOv12 Object Detection</h1>
            
            <div style={{ background: 'white', padding: '20px', borderRadius: '8px', marginBottom: '20px' }}>
                <div>
                    <label>Image File:</label>
                    <input type="file" onChange={(e) => setFile(e.target.files[0])} accept="image/*" />
                </div>
                
                <div>
                    <label>Or Image URL:</label>
                    <input type="url" value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://example.com/image.jpg" />
                </div>
                
                <div>
                    <label>Model: <span>{model}</span></label>
                    <select value={model} onChange={(e) => setModel(e.target.value)}>
                        <option value="yolov12n.pt">YOLOv12 Nano</option>
                        <option value="yolov12s.pt">YOLOv12 Small</option>
                        <option value="yolov12m.pt">YOLOv12 Medium</option>
                        <option value="yolov12l.pt">YOLOv12 Large</option>
                        <option value="yolov12x.pt">YOLOv12 XLarge</option>
                    </select>
                </div>
                
                <div>
                    <label>Confidence: {confidence.toFixed(2)}</label>
                    <input type="range" min="0" max="1" step="0.05" value={confidence} onChange={(e) => setConfidence(parseFloat(e.target.value))} />
                </div>
                
                <div>
                    <label>Image Size: {imageSize}</label>
                    <input type="range" min="320" max="1280" step="32" value={imageSize} onChange={(e) => setImageSize(parseInt(e.target.value))} />
                </div>
                
                <button onClick={predict} disabled={loading}>
                    {loading ? 'Detecting...' : 'Detect Objects'}
                </button>
                
                {error && <p style={{ color: 'red', marginTop: '10px' }}>{error}</p>}
            </div>
            
            {results && (
                <div style={{ background: 'white', padding: '20px', borderRadius: '8px' }}>
                    <h2>Results</h2>
                    <p><strong>Detections:</strong> {results.detection_count}</p>
                    <p><strong>Model:</strong> {results.model}</p>
                    
                    {results.detection_count > 0 && (
                        <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: '20px' }}>
                            <thead>
                                <tr style={{ background: '#f8f9fa' }}>
                                    <th style={{ padding: '10px', textAlign: 'left' }}>Class</th>
                                    <th style={{ padding: '10px', textAlign: 'left' }}>Confidence</th>
                                    <th style={{ padding: '10px', textAlign: 'left' }}>BBox</th>
                                </tr>
                            </thead>
                            <tbody>
                                {results.detections.map((det, idx) => (
                                    <tr key={idx} style={{ borderBottom: '1px solid #ddd' }}>
                                        <td style={{ padding: '10px' }}>{det.class_name}</td>
                                        <td style={{ padding: '10px' }}>{(det.confidence * 100).toFixed(1)}%</td>
                                        <td style={{ padding: '10px' }}>({det.bbox.x1.toFixed(0)}, {det.bbox.y1.toFixed(0)})</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    )}
                </div>
            )}
        </div>
    );
};

export default YOLOv12Detector;
*/

// ============================================================
// 4. Usage in Node.js
// ============================================================

/*
const client = new YOLOv12APIClient('http://localhost:8000');

(async () => {
    try {
        // Check health
        const health = await client.healthCheck();
        console.log('API Health:', health);
        
        // Get available models
        const models = await client.getAvailableModels();
        console.log('Available Models:', models);
        
        // Predict from URL
        const result = await client.predictFromUrl(
            'https://ultralytics.com/images/bus.jpg',
            'yolov12m.pt',
            0.25,
            640
        );
        
        console.log(`Found ${result.detection_count} objects:`);
        result.detections.forEach(det => {
            console.log(`  - ${det.class_name}: ${(det.confidence * 100).toFixed(1)}%`);
        });
    } catch (error) {
        console.error('Error:', error);
    }
})();
*/
