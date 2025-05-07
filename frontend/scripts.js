document.getElementById("predict-btn").addEventListener("click", async (event) => {
    event.preventDefault();

    const uploadInput = document.getElementById("upload");
    const predictionResult = document.getElementById("prediction-result");

    predictionResult.textContent = "Processing...";

    const files = uploadInput.files;
    if (files.length === 0) {
        alert ("Please upload at least one PNG image!");
        predictionResult.textContent = "No result yet.";
        return;
    }

    const formData = new FormData();
    for (const file of files) {
        formData.append("files", file);
    }

    try {
        const response = await fetch("http://127.0.0.1:8000/predict-cancer/", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();

        if (data.overall_cancer_detected) {
            console.log("Cancer detected!");
            predictionResult.textContent = "Cancer detected in some images!";
        } else {
            console.log("No cancer detected.");
            predictionResult.textContent = "No cancer detected in any images.";
        }

        // Log individual file predictions for debugging
        console.log(data.predictions);

        // Update the UI to reflect individual results
        data.predictions.forEach((prediction) => {
            if (prediction.has_cancer) {
                console.log(`Cancer detected in: ${prediction.filename}`);
            }
        });

    } catch (error) {
        predictionResult.textContent = `Error: ${error.message}`;
    }
    return false;  // Prevent page reload or form submission
});
