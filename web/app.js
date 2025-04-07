document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    const clearButton = document.getElementById('clear');
    const predictButton = document.getElementById('predict');
    const resultDiv = document.getElementById('result');

    // Set canvas drawing style
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    canvas.style.backgroundColor = 'black';

    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;

    // Drawing functions
    function startDrawing(e) {
        isDrawing = true;
        [lastX, lastY] = getCoordinates(e);
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault();

        const [currentX, currentY] = getCoordinates(e);

        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(currentX, currentY);
        ctx.stroke();

        [lastX, lastY] = [currentX, currentY];
    }

    function stopDrawing() {
        isDrawing = false;
    }

    function getCoordinates(e) {
        let x, y;
        if (e.touches) {
            x = e.touches[0].clientX - canvas.getBoundingClientRect().left;
            y = e.touches[0].clientY - canvas.getBoundingClientRect().top;
        } else {
            x = e.offsetX;
            y = e.offsetY;
        }
        return [x, y];
    }

    // Event listeners for mouse and touch
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    canvas.addEventListener('touchstart', startDrawing);
    canvas.addEventListener('touchmove', draw);
    canvas.addEventListener('touchend', stopDrawing);

    // Clear canvas
    function clearCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        resultDiv.textContent = 'Prediction: -';
    }

    clearButton.addEventListener('click', clearCanvas);

    // Preprocess the canvas image to match MNIST format
    function preprocessImage() {
        // Create a temporary canvas for preprocessing
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = 28;
        tempCanvas.height = 28;

        // Scale down to 28x28 and get image data
        tempCtx.drawImage(canvas, 0, 0, 28, 28);
        const imageData = tempCtx.getImageData(0, 0, 28, 28);
        const data = imageData.data;

        // Convert to grayscale float32 array and normalize
        const input = new Float32Array(28 * 28);
        for (let i = 0; i < data.length; i += 4) {
            input[i / 4] = data[i] / 255.0; // Only need red channel since it's grayscale
        }

        return input;
    }

    // Load and run the ONNX model
    async function loadModel() {
        try {
            const session = await ort.InferenceSession.create('model.onnx');
            return session;
        } catch (e) {
            console.error('Failed to load model:', e);
            return null;
        }
    }

    let model = null;
    loadModel().then(loadedModel => {
        model = loadedModel;
        predictButton.disabled = false;
    });

    // Predict function
    async function predict() {
        if (!model) {
            resultDiv.textContent = 'Model not loaded yet';
            return;
        }

        try {
            const input = preprocessImage();
            const tensor = new ort.Tensor('float32', input, [1, 1, 28, 28]);
            const results = await model.run({ input: tensor });
            const output = results.output.data;

            // Get the predicted digit (index of maximum value)
            const predictedDigit = output.indexOf(Math.max(...output));
            resultDiv.textContent = `Prediction: ${predictedDigit}`;
        } catch (e) {
            console.error('Prediction failed:', e);
            resultDiv.textContent = 'Prediction failed';
        }
    }

    predictButton.addEventListener('click', predict);
});
