// static/main.js

document.addEventListener("DOMContentLoaded", async () => {
    // Fetch current level from backend
    let currentUserLevel = 1; // default fallback
    try {
      const response = await fetch('/get_current_level');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      currentUserLevel = parseInt(data.level);
      
      // Fetch and display the initial prompt for the current level
      const promptResponse = await fetch(`/get_prompt?key=${currentUserLevel}`);
      if (promptResponse.ok) {
        const promptData = await promptResponse.json();
        const promptDisplayElement = document.querySelector(".prompt-box");
        if (promptDisplayElement) {
          promptDisplayElement.innerText = promptData.prompt_text;
        }
      }
    } catch (error) {
      console.warn("Failed to fetch current level:", error);
    }
   
     // Update progress bar based on currentUserLevel
     const maxLevel = 3;
     const progressPercentage = ((currentUserLevel - 1) / maxLevel) * 100;
     const progressBar = document.querySelector('.progress-bar');
     if (progressBar) {
         progressBar.style.width = `${progressPercentage}%`;
         progressBar.textContent = `${currentUserLevel - 1}/${maxLevel}`;
     }

    
    // Define the pastel color palette for token visualizations.
    const defaultTokenColors = [
      "#FFD1DC", // pastel pink
      "#C5E3BF", // pastel green
      "#FFFFCC", // pastel yellow
      "#FFEBCC"  // pastel peach
    ];
  
    // --- Toast Function Implementation ---
    function showToast(message, type = "info") {
      const toast = document.createElement("div");
      toast.className = `toast ${type}`;
      toast.innerText = message;
      document.body.appendChild(toast);
      setTimeout(() => {
        toast.classList.add("fade-out");
        setTimeout(() => {
          document.body.removeChild(toast);
        }, 500);
      }, 3000);
    }
    // --- End Toast Function Implementation ---
 
    // After fetching currentUserLevel and updating the progress bar:
    const levelSelect = document.getElementById("systemPromptChoice");
    if (levelSelect) {
    // Loop through options to enable/disable based on currentUserLevel.
    for (let option of levelSelect.options) {
        const optionLevel = parseInt(option.value);
        if (optionLevel <= currentUserLevel) {
        option.disabled = false;
        option.textContent = option.textContent.replace(" 🔒", "");
        } else {
        option.disabled = true;
        if (!option.textContent.includes("🔒")) {
            option.textContent += " 🔒";
        }
        }
    }
    // Automatically select the option matching the current user's level.
    levelSelect.value = currentUserLevel.toString();
    } else {
    console.warn("levelSelect is missing.");
    }
  
    // Get references to other key DOM elements.
    const submitBtnElem = document.getElementById("submitBtn");
    const modelChoiceElem = document.getElementById("modelChoice");
  
    if (!submitBtnElem) {
      console.error("Element with id 'submitBtn' not found.");
      return;
    }
    if (!modelChoiceElem) {
      console.error("Element with id 'modelChoice' not found.");
      return;
    }
  
    // When the level selection changes, fetch the corresponding system prompt text.
    levelSelect.addEventListener("change", async (event) => {
      const selectedKey = event.target.value;
      try {
        const response = await fetch(`/get_prompt?key=${selectedKey}`);
        if (!response.ok) {
          throw new Error(`Server returned ${response.status}`);
        }
        const data = await response.json();
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
  
    // Event listener for model selection change (unchanged).
    modelChoiceElem.addEventListener("change", async (event) => {
      const selectedUUID = event.target.value;
      const modelText = event.target.options[event.target.selectedIndex].text;
      const spinner = document.getElementById("modelSpinner");
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
  
    // Main submit handler for text generation
    submitBtnElem.addEventListener("click", async () => {
      const userPrompt = document.getElementById("userPrompt").value;
      const maxTokens = parseInt(document.getElementById("maxTokens").value) || 100;
      const modelChoice = document.getElementById("modelChoice").value;
      const systemPromptChoice = document.getElementById("systemPromptChoice").value;

      showToast("Generating text...", "info");
      console.log("Submit button pressed. Calling /generate endpoint...");

      const payload = {
        user_prompt: userPrompt,
        max_new_tokens: maxTokens,
        model_choice: modelChoice,
        system_prompt_choice: `level-${systemPromptChoice}`
      };
  
      document.getElementById("displayCombinedPrompt").innerText = "";
      document.getElementById("displayOutputText").innerText = "";
      document.getElementById("userTokenDisplay").innerHTML = "";
      document.getElementById("outputTokenDisplay").innerHTML = "";
  
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
  
        spinner.classList.add("hidden");
        document.getElementById("displayCombinedPrompt").innerText = data.combined_prompt;
        document.getElementById("displayOutputText").innerText = data.generated_text_only;
  
        const userTokenDisplay = document.getElementById("userTokenDisplay");
        data.user_tokens.forEach((tokenText, index) => {
          const tokenSpan = document.createElement("span");
          tokenSpan.classList.add("token");
          tokenSpan.style.backgroundColor = defaultTokenColors[index % defaultTokenColors.length];
          tokenSpan.title = `Token ${index}: ${tokenText}`;
          tokenSpan.textContent = tokenText;
          userTokenDisplay.appendChild(tokenSpan);
        });
  
        const outputTokenDisplay = document.getElementById("outputTokenDisplay");
        data.output_tokens.forEach((tokenText, index) => {
          const tokenSpan = document.createElement("span");
          tokenSpan.classList.add("token");
          tokenSpan.style.backgroundColor = defaultTokenColors[index % defaultTokenColors.length];
          tokenSpan.title = `Token ${index}: ${tokenText}`;
          tokenSpan.textContent = tokenText;
          outputTokenDisplay.appendChild(tokenSpan);
        });
        
        showToast("Text generated successfully!", "success");
  
        if (data.jailbreak_success) {
          confetti({
            particleCount: 150,
            spread: 70,
            origin: { y: 0.6 }
          });
          showToast("another morbilly in the bank", "success");
        }
  
      } catch (error) {
        console.error("Error during fetch:", error);
        document.getElementById("displayOutputText").innerText = "An error occurred: " + error;
        spinner.classList.add("hidden");
        showToast("Error generating text", "error");
      }
    });
  });
