document.addEventListener("DOMContentLoaded", async () => {
    // Default fallbacks
    let currentLevel = 1;   // The level the user wants to play
    let highestLevel = 1;   // The maximum level the user has cleared
    let maxLevel = 1;       // The maximum level available in the system
    let levelsData = [];    // To hold the level objects fetched from backend

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

      // Add new fetch for levels data
      const levelsResponse = await fetch('/levels');
      if (levelsResponse.ok) {
          const levelsJson = await levelsResponse.json();
          levelsData = levelsJson.levels;
          maxLevel = levelsJson.total_levels;  // Set maxLevel from the total_levels value
      } else {
          console.warn("Failed to fetch levels data from /levels");
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

    // Add difficulty colors definition
    const difficultyColors = {
      "Easy": "#28a745",    // green
      "Medium": "#fd7e14",  // orange
      "Hard": "#dc3545",    // red
      "Completed": "#9b59b6" // purple
    };
  
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
 
    // Build custom level selection boxes
    const levelContainer = document.getElementById("levelSelectionContainer");
    const descriptionBox = document.getElementById("description-box"); // Reference the description-box element
    if (levelContainer) {
        // Clear any existing content
        levelContainer.innerHTML = '';
        // Sort levels by their index
        levelsData.sort((a, b) => a.index - b.index);
        // Create a box for each level
        levelsData.forEach(level => {
            const levelBox = document.createElement("div");
            levelBox.className = "level-box";
            // Build inner HTML with a difficulty badge inside a box
            if (level.index > highestLevel) {
                levelBox.classList.add("disabled");
                levelBox.innerHTML = `
                  <div class="level-name">${level.name}</div>
                  <div class="difficulty-box" style="background-color: gray;">
                    Difficulty: Unknown 🔒
                  </div>`;
            } else {
                const diffColor = level.index < currentLevel ? 
                    difficultyColors["Completed"] : 
                    difficultyColors[level.difficulty] || "#ccc";
                levelBox.innerHTML = `
                  <div class="level-name">${level.name}</div>
                  <div class="difficulty-box" style="background-color: ${diffColor};">
                    ${level.difficulty}
                  </div>`;
                if (level.index === currentLevel) {
                    levelBox.classList.add("selected");
                    // Populate the description-box with the current level's description
                    if (descriptionBox) {
                        descriptionBox.innerText = level.description;
                    }
                } else {
                    // Add click handler to update the description-box and change the level
                    levelBox.addEventListener("click", async () => {
                        try {
                            const response = await fetch("/update_level", {
                                method: "POST",
                                headers: {"Content-Type": "application/json"},
                                body: JSON.stringify({ new_level: level.index })
                            });
                            if (!response.ok) {
                                throw new Error(`Server returned ${response.status}`);
                            }
                            // Update the description-box with the selected level's description
                            if (descriptionBox) {
                                descriptionBox.innerText = level.description;
                            }
                            // Reload the page so that a new conversation is started.
                            window.location.reload();
                        } catch (error) {
                            console.error("Error updating level:", error);
                            showToast("Error updating level", "error");
                        }
                    });
                }
            }
            levelContainer.appendChild(levelBox);
        });
    }
  
    // Add toggle functionality for level selection
    const levelHeader = document.getElementById("levelSelectionHeader");
    const toggleIcon = document.getElementById("levelToggleIcon");
    if (levelHeader && levelContainer && toggleIcon) {
        // Get saved state from localStorage, default to collapsed (false)
        const isExpanded = localStorage.getItem('levelSelectExpanded') === 'true';
        
        // Set initial state based on saved preference
        toggleIcon.innerHTML = isExpanded 
            ? '<i class="fa-solid fa-chevron-down"></i>'
            : '<i class="fa-solid fa-chevron-up"></i>';
        
        if (isExpanded) {
            levelContainer.classList.add('active');
        }
        
        levelHeader.addEventListener("click", () => {
            levelContainer.classList.toggle('active');
            const isNowExpanded = levelContainer.classList.contains('active');
            
            // Save state to localStorage
            localStorage.setItem('levelSelectExpanded', isNowExpanded);
            
            toggleIcon.innerHTML = isNowExpanded
                ? '<i class="fa-solid fa-chevron-down"></i>'
                : '<i class="fa-solid fa-chevron-up"></i>';
        });
    }

    // Add chat window resizer
    const chatHistory = document.getElementById("chatHistory");
    const chatResizer = document.getElementById("chatResizer");
    if (chatHistory && chatResizer) {
        let isResizing = false;
        chatResizer.addEventListener("mousedown", (e) => {
            isResizing = true;
            document.body.style.cursor = "ns-resize";
        });
        document.addEventListener("mousemove", (e) => {
            if (!isResizing) return;
            const newHeight = Math.max(100, e.clientY - chatHistory.getBoundingClientRect().top);
            chatHistory.style.height = `${newHeight}px`;
        });
        document.addEventListener("mouseup", () => {
            if (isResizing) {
                isResizing = false;
                document.body.style.cursor = "default";
            }
        });
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
            const maxTokens = parseInt(document.getElementById('maxTokens').value) || 100;
            
            const payload = {
                user_prompt: userMessage,
                max_new_tokens: maxTokens,
                model_choice: modelChoice,
                system_prompt_choice: `level-${currentLevel}`
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
                
                // Handle jailbreak detection
                if (data.jailbreak_detected) {
                    showToast("Jailbreak detected!", "failure");
                    waitingBubble.remove();
                    return;
                }

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
                    showToast("Congratulations! Successful jailbreak detected.", "success");
                    
                    // Add countdown toast
                    let countdown = 5;
                    const countdownInterval = setInterval(() => {
                        showToast(`Page will reset in ${countdown - 1} seconds...`, "info");
                        countdown--;
                        if (countdown < 0) {
                            clearInterval(countdownInterval);
                        }
                    }, 1000);

                    // Add page refresh after 5 seconds
                    setTimeout(() => {
                        window.location.reload();
                    }, 5000);
                }
            } catch (error) {
                // Remove waiting bubble if error occurs
                waitingBubble.remove();
                console.error("Error during generation:", error);
                showToast("Error generating text", "error");
            }
        });
    }
  
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