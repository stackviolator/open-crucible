// static/main.js

// Updated pastel color palette for tokens, blue is reserved for hover
const defaultTokenColors = [
    "#FFD1DC", // pastel pink
    "#C5E3BF", // pastel green
    "#FFFFCC", // pastel yellow
    "#FFEBCC"  // pastel peach
];

document.getElementById("submitBtn").addEventListener("click", async () => {
    const userPrompt = document.getElementById("userPrompt").value;
    const maxTokens = parseInt(document.getElementById("maxTokens").value) || 100;

    console.log("Submit button pressed. Calling /generate endpoint...");

    const payload = {
        user_prompt: userPrompt,
        max_new_tokens: maxTokens
    };

    // Clear previous outputs
    document.getElementById("displayCombinedPrompt").innerText = "";
    document.getElementById("displayOutputText").innerText = "";
    document.getElementById("userTokenDisplay").innerHTML = "";
    document.getElementById("outputTokenDisplay").innerHTML = "";

    // Show spinner
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

        // Hide spinner when data is received
        spinner.classList.add("hidden");

        // Display combined prompt and model output
        document.getElementById("displayCombinedPrompt").innerText = data.combined_prompt;
        document.getElementById("displayOutputText").innerText = data.generated_text_only;
        
        // Build token visualization for user tokens:
        const userTokenDisplay = document.getElementById("userTokenDisplay");
        data.user_tokens.forEach((tokenText, index) => {
            const tokenSpan = document.createElement("span");
            tokenSpan.classList.add("token");
            // Use the pastel color palette for background
            tokenSpan.style.backgroundColor = defaultTokenColors[index % defaultTokenColors.length];
            tokenSpan.title = `Token ${index}: ${tokenText}`;
            tokenSpan.textContent = tokenText;
            userTokenDisplay.appendChild(tokenSpan);
        });

        // Build token visualization for output tokens:
        const outputTokenDisplay = document.getElementById("outputTokenDisplay");
        data.output_tokens.forEach((tokenText, index) => {
            const tokenSpan = document.createElement("span");
            tokenSpan.classList.add("token");
            tokenSpan.style.backgroundColor = defaultTokenColors[index % defaultTokenColors.length];
            tokenSpan.title = `Token ${index}: ${tokenText}`;
            tokenSpan.textContent = tokenText;
            outputTokenDisplay.appendChild(tokenSpan);
        });

    } catch (error) {
        console.error("Error during fetch:", error);
        document.getElementById("displayOutputText").innerText = "An error occurred: " + error;
        spinner.classList.add("hidden");
    }
});
