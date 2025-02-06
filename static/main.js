// static/main.js

document.addEventListener("DOMContentLoaded", () => {
    // Define the pastel color palette for token visualizations.
    // Blue is reserved for hover effects.
    const defaultTokenColors = [
        "#FFD1DC", // pastel pink
        "#C5E3BF", // pastel green
        "#FFFFCC", // pastel yellow
        "#FFEBCC"  // pastel peach
    ];

    // --- Toast Function Implementation ---
    /**
     * Creates a toast message that pops up and then fades out.
     * @param {string} message - The message to display.
     * @param {string} type - The type of message ("success", "error", "info") for styling.
     */
    function showToast(message, type = "info") {
        const toast = document.createElement("div");
        toast.className = `toast ${type}`; // Use CSS classes for styling
        toast.innerText = message;
        document.body.appendChild(toast);
        // Automatically fade out after 3 seconds, then remove from DOM
        setTimeout(() => {
            toast.classList.add("fade-out");
            setTimeout(() => {
                document.body.removeChild(toast);
            }, 500);
        }, 3000);
    }
    // --- End Toast Function Implementation ---

    // Get references to the key DOM elements.
    const systemPromptChoiceElem = document.getElementById("systemPromptChoice");
    const submitBtnElem = document.getElementById("submitBtn");
    const modelChoiceElem = document.getElementById("modelChoice");  // Fixed ID reference

    // Ensure these elements exist.
    if (!systemPromptChoiceElem) {
        console.error("Element with id 'systemPromptChoice' not found.");
        return;
    }
    if (!submitBtnElem) {
        console.error("Element with id 'submitBtn' not found.");
        return;
    }
    if (!modelChoiceElem) {
        console.error("Element with id 'modelChoice' not found.");
        return;
    }

    // Event listener: when the user changes the system prompt selection,
    // fetch the corresponding prompt text from the server.
    systemPromptChoiceElem.addEventListener("change", async (event) => {
        const selectedKey = event.target.value;
        // Show a toast that the system prompt is being updated.
        showToast("Changing system prompt...", "info");
        try {
            // Make an HTTP GET request to the /get_prompt endpoint.
            const response = await fetch(`/get_prompt?key=${encodeURIComponent(selectedKey)}`);
            if (!response.ok) {
                throw new Error(`Server returned ${response.status}`);
            }
            const data = await response.json();
            // Update the prompt display area (element with class "prompt-box") with the fetched text.
            const promptDisplayElement = document.querySelector(".prompt-box");
            if (promptDisplayElement) {
                promptDisplayElement.innerText = data.prompt_text;
            } else {
                console.error("Element with class 'prompt-box' not found.");
            }
        } catch (error) {
            console.error("Error fetching system prompt text:", error);
        }
    });

    // Event listener: when the user changes the model selection,
    // fetch the corresponding model name from the server.
    modelChoiceElem.addEventListener("change", async (event) => {
        const selectedUUID = event.target.value;
        // Get the human-readable model name from the selected option's text.
        const modelText = event.target.options[event.target.selectedIndex].text;

        // Get spinner element from the DOM.
        const spinner = document.getElementById("modelSpinner");
        
        // Show the spinner and display a toast message.
        if (spinner) spinner.classList.remove("hidden");
        showToast("Changing model...", "info");

        try {
            const response = await fetch("/change_model", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model_uuid: selectedUUID })
            });
            const data = await response.json();
            
            if (data.status === "success") {
                showToast("Model changed to " + data.model_name, "success");
                console.log("Model changed to", data.model_name);
            } else {
                // Show a generic error message.
                showToast("Error changing model to: " + modelText, "error");
                console.error("Error changing model:", data.error);
            }
        } catch (error) {
            showToast("Error changing model to: " + modelText, "error");
            console.error("Error during model change request:", error);
        } finally {
            if (spinner) spinner.classList.add("hidden");
        }
    });

    // Main submit handler for the generate button.
    submitBtnElem.addEventListener("click", async () => {
        // Get the values from input elements.
        const userPrompt = document.getElementById("userPrompt").value;
        const maxTokens = parseInt(document.getElementById("maxTokens").value) || 100;
        const systemPromptChoice = document.getElementById("systemPromptChoice").value;
        const modelChoice = document.getElementById("modelChoice").value; // Correct element ID

        // Show a toast to indicate generation has started.
        showToast("Generating text...", "info");

        console.log("Submit button pressed. Calling /generate endpoint...");

        // Construct the payload to send to the /generate endpoint.
        const payload = {
            user_prompt: userPrompt,
            max_new_tokens: maxTokens,
            system_prompt_choice: systemPromptChoice,
            model_choice: modelChoice
        };

        // Clear previous outputs.
        document.getElementById("displayCombinedPrompt").innerText = "";
        document.getElementById("displayOutputText").innerText = "";
        document.getElementById("userTokenDisplay").innerHTML = "";
        document.getElementById("outputTokenDisplay").innerHTML = "";

        // Show spinner.
        const spinner = document.getElementById("spinner");
        spinner.classList.remove("hidden");

        try {
            const response = await fetch("/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            console.log("Response received.");
            const data = await response.json();

            // Hide spinner.
            spinner.classList.add("hidden");

            // Display combined prompt and model output.
            document.getElementById("displayCombinedPrompt").innerText = data.combined_prompt;
            document.getElementById("displayOutputText").innerText = data.generated_text_only;

            // Visualize user tokens.
            const userTokenDisplay = document.getElementById("userTokenDisplay");
            data.user_tokens.forEach((tokenText, index) => {
                const tokenSpan = document.createElement("span");
                tokenSpan.classList.add("token");
                tokenSpan.style.backgroundColor = defaultTokenColors[index % defaultTokenColors.length];
                tokenSpan.title = `Token ${index}: ${tokenText}`;
                tokenSpan.textContent = tokenText;
                userTokenDisplay.appendChild(tokenSpan);
            });

            // Visualize output tokens.
            const outputTokenDisplay = document.getElementById("outputTokenDisplay");
            data.output_tokens.forEach((tokenText, index) => {
                const tokenSpan = document.createElement("span");
                tokenSpan.classList.add("token");
                tokenSpan.style.backgroundColor = defaultTokenColors[index % defaultTokenColors.length];
                tokenSpan.title = `Token ${index}: ${tokenText}`;
                tokenSpan.textContent = tokenText;
                outputTokenDisplay.appendChild(tokenSpan);
            });
            
            // Show success toast on generation complete.
            showToast("Text generated successfully!", "success");
        } catch (error) {
            console.error("Error during fetch:", error);
            document.getElementById("displayOutputText").innerText = "An error occurred: " + error;
            spinner.classList.add("hidden");
            showToast("Error generating text", "error");
        }
    });
});
