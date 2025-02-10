// static/main.js

document.addEventListener("DOMContentLoaded", async () => {
    // Default fallbacks
    let currentLevel = 1;   // The level the user wants to play
    let highestLevel = 1;   // The maximum level the user has cleared
    let maxLevel = 3;       // The maximum level available in the system

    try {
      // Fetch both current and highest level from the server
      const levelResponse = await fetch('/get_current_level');
      if (levelResponse.ok) {
        const levelData = await levelResponse.json();
        // Add validation for current_level and highest_level
        currentLevel = parseInt(levelData.current_level);
        highestLevel = parseInt(levelData.highest_level);
      }
      
      // Fetch configuration, including the system's max level
      const configResponse = await fetch('/config');
      if (configResponse.ok) {
        const configData = await configResponse.json();
        maxLevel = parseInt(configData.max_level);
      }

      // Fetch and display the initial prompt for the current level
      const promptResponse = await fetch(`/get_prompt?key=${currentLevel}`);
      if (promptResponse.ok) {
        const promptData = await promptResponse.json();
        const promptDisplayElement = document.querySelector(".prompt-box");
        if (promptDisplayElement) {
          promptDisplayElement.innerText = promptData.prompt_text;
        }
      }
    } catch (error) {
      console.warn("Failed to fetch level information:", error);
    }

    // Update the progress bar using highestLevel (to reflect cleared progress)
    const progressBarContainer = document.querySelector('.progress-bar-container');
    const progressPercentage = ((highestLevel - 1) / maxLevel) * 100;
    const progressBar = document.querySelector('.progress-bar');
    if (progressBar && progressBarContainer) {
      progressBar.style.width = `${progressPercentage}%`;
      progressBarContainer.setAttribute('data-text', `${highestLevel - 1}/${maxLevel}`);
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
 
    // After fetching levels, update the level selection dropdown
    const levelSelect = document.getElementById("systemPromptChoice");
    if (levelSelect) {
      // Enable options up to highestLevel and grey out the others
      for (let option of levelSelect.options) {
        const optionLevel = parseInt(option.value);
        if (optionLevel <= highestLevel) {
          option.disabled = false;
          option.textContent = option.textContent.replace(" 🔒", "");
        } else {
          option.disabled = true;
          if (!option.textContent.includes("🔒")) {
            option.textContent += " 🔒";
          }
        }
      }
      // Automatically select the option matching the current_level
      levelSelect.value = currentLevel.toString();
    } else {
      console.warn("levelSelect is missing.");
    }
  
    // Get references to other key DOM elements.
    const modelChoiceElem = document.getElementById("modelChoice");
  
    if (!modelChoiceElem) {
      console.error("Element with id 'modelChoice' not found.");
      return;
    }

    // Load chat history
    loadChatHistory();

    // Get reference to chat input
    const chatInput = document.getElementById('chatInput');

    // Add enter key handler for chat input
    if (chatInput) {
        chatInput.addEventListener('keypress', async (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault(); // Prevent default enter behavior
                chatSendBtn.click(); // Trigger the same click handler
            }
        });
    }

    // Chat send button
    const chatSendBtn = document.getElementById('chatSendBtn');
    if (chatSendBtn) {
        chatSendBtn.addEventListener('click', async () => {
            const chatInput = document.getElementById('chatInput');
            const userMessage = chatInput.value.trim();
            if (!userMessage) return;
            
            // Append the user's message to the chat history
            const chatHistoryElement = document.getElementById('chatHistory');
            const userBubble = document.createElement('div');
            userBubble.className = 'chat-message user';
            userBubble.textContent = userMessage;
            chatHistoryElement.appendChild(userBubble);
            
            // Get model settings
            const modelChoice = document.getElementById("modelChoice").value;
            const systemPromptChoice = document.getElementById('systemPromptChoice').value;
            const maxTokens = parseInt(document.getElementById('maxTokens').value) || 100;
            
            const payload = {
                user_prompt: userMessage,
                max_new_tokens: maxTokens,
                model_choice: modelChoice,
                system_prompt_choice: `level-${systemPromptChoice}`
            };

            // Clear input and show loading state
            chatInput.value = '';
            showToast("Generating text...", "info");

            // --- Add waiting bubble with animated dots ---
            const waitingBubble = document.createElement('div');
            waitingBubble.className = 'chat-message assistant waiting';
            waitingBubble.innerHTML = '<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span>';
            chatHistoryElement.appendChild(waitingBubble);
            chatHistoryElement.scrollTop = chatHistoryElement.scrollHeight;
            
            try {
                const response = await fetch("/generate", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
                
                const data = await response.json();
                
                // Remove the waiting bubble before displaying the real response
                waitingBubble.remove();
                
                // Append the assistant's reply as a chat bubble
                const assistantBubble = document.createElement('div');
                assistantBubble.className = 'chat-message assistant';
                assistantBubble.textContent = data.generated_text_only;
                chatHistoryElement.appendChild(assistantBubble);
                chatHistoryElement.scrollTop = chatHistoryElement.scrollHeight;
                
                // Optionally update additional info elements if they exist
                const combinedPromptElem = document.getElementById("displayCombinedPrompt");
                if (combinedPromptElem) {
                    combinedPromptElem.innerText = data.combined_prompt;
                }
                const outputTextElem = document.getElementById("displayOutputText");
                if (outputTextElem) {
                    outputTextElem.innerText = data.generated_text_only;
                }
                const userTokenDisplay = document.getElementById("userTokenDisplay");
                if (userTokenDisplay) {
                    userTokenDisplay.innerHTML = "";
                    data.user_tokens.forEach((tokenText, index) => {
                        const tokenSpan = document.createElement("span");
                        tokenSpan.classList.add("token");
                        tokenSpan.style.backgroundColor = defaultTokenColors[index % defaultTokenColors.length];
                        tokenSpan.title = `Token ${index}: ${tokenText}`;
                        tokenSpan.textContent = tokenText;
                        userTokenDisplay.appendChild(tokenSpan);
                    });
                }
                const outputTokenDisplay = document.getElementById("outputTokenDisplay");
                if (outputTokenDisplay) {
                    outputTokenDisplay.innerHTML = "";
                    data.output_tokens.forEach((tokenText, index) => {
                        const tokenSpan = document.createElement("span");
                        tokenSpan.classList.add("token");
                        tokenSpan.style.backgroundColor = defaultTokenColors[index % defaultTokenColors.length];
                        tokenSpan.title = `Token ${index}: ${tokenText}`;
                        tokenSpan.textContent = tokenText;
                        outputTokenDisplay.appendChild(tokenSpan);
                    });
                }
                
                showToast("Text generated successfully!", "success");
      
                // Handle jailbreak success if present
                if (data.jailbreak_success) {
                    confetti({
                        particleCount: 150,
                        spread: 70,
                        origin: { y: 0.6 }
                    });
                    showToast("another morbilly in the bank", "success");
                }
            } catch (error) {
                // Remove waiting bubble if error occurs
                waitingBubble.remove();
                console.error("Error during generation:", error);
                showToast("Error generating text", "error");
            }
        });
    }
  
    // When the level selection changes, update the level and reset the conversation.
    levelSelect.addEventListener("change", async (event) => {
    const newLevel = event.target.value;
    try {
        const response = await fetch("/update_level", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ new_level: parseInt(newLevel) })
        });
        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }
        // Reload the page so that a new conversation is started.
        window.location.reload();
    } catch (error) {
        console.error("Error updating level:", error);
        showToast("Error updating level", "error");
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
  });

async function logout() {
    try {
        const response = await fetch('/logout', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (response.ok) {
            window.location.href = '/login';
        } else {
            showToast('Logout failed', 'error');
        }
    } catch (error) {
        console.error('Error during logout:', error);
        showToast('Logout failed', 'error');
    }
}

// Function to load the chat history
async function loadChatHistory() {
  try {
      const response = await fetch('/chat_history');
      if (response.ok) {
          const chatData = await response.json();
          const chatHistoryElement = document.getElementById('chatHistory');
          if (chatHistoryElement) {
              // Clear any existing content
              chatHistoryElement.innerHTML = '';
              chatData.forEach(chatItem => {
                  // If there is a user prompt, display it
                  if (chatItem.user) {
                      const userBubble = document.createElement('div');
                      userBubble.className = 'chat-message user';
                      userBubble.textContent = chatItem.user;
                      chatHistoryElement.appendChild(userBubble);
                  }
                  // If there is an assistant reply, display it
                  if (chatItem.assistant) {
                      const assistantBubble = document.createElement('div');
                      assistantBubble.className = 'chat-message assistant';
                      assistantBubble.textContent = chatItem.assistant;
                      chatHistoryElement.appendChild(assistantBubble);
                  }
              });
              // Optional: scroll to the bottom of the chat history
              chatHistoryElement.scrollTop = chatHistoryElement.scrollHeight;
          }
      } else {
          console.error('Failed to fetch chat history. Status:', response.status);
      }
  } catch (error) {
      console.error('Error loading chat history:', error);
  }
}