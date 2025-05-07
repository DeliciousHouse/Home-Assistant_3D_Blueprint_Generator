// Blueprint Generator Web UI JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // App state
    let currentFloor = 0;
    let blueprint = null;
    let unitSystem = 'metric'; // Default unit system: 'metric' or 'imperial'
    let currentView = 'standard'; // Default view mode

    // UI Elements
    const generateBtn = document.getElementById('generate-btn');
    const floorUpBtn = document.getElementById('floor-up');
    const floorDownBtn = document.getElementById('floor-down');
    const currentFloorIndicator = document.getElementById('current-floor');
    const toggleUnitSystemBtn = document.getElementById('toggle-unit-system');
    const toggleRoomLabelsBtn = document.getElementById('toggle-room-labels');
    const toggleObjectLabelsBtn = document.getElementById('toggle-object-labels');
    const toggleDimensionsBtn = document.getElementById('toggle-dimensions');
    const standardViewBtn = document.getElementById('standard-view');
    const measurementViewBtn = document.getElementById('measurement-view');
    const furnitureViewBtn = document.getElementById('furniture-view');
    const blueprintCanvas = document.getElementById('blueprint-canvas');
    const loadingSpinner = document.getElementById('loading-spinner');
    const statusMessage = document.getElementById('status-message');
    const canvasContainer = document.querySelector('.canvas-container');

    // Initialize UI state
    initializeUI();
    loadBlueprint();

    // Event Listeners
    if (floorUpBtn) {
        floorUpBtn.addEventListener('click', () => changeFloor(1));
    }

    if (floorDownBtn) {
        floorDownBtn.addEventListener('click', () => changeFloor(-1));
    }

    if (generateBtn) {
        generateBtn.addEventListener('click', generateBlueprint);
    }

    if (toggleUnitSystemBtn) {
        // Set initial button text
        toggleUnitSystemBtn.innerHTML = `
            <span class="material-icons">straighten</span>
            ${unitSystem === 'metric' ? 'Metric' : 'Imperial'}
        `;

        toggleUnitSystemBtn.addEventListener('click', toggleUnitSystem);
    }

    // Toggle buttons event listeners
    if (toggleRoomLabelsBtn) {
        toggleRoomLabelsBtn.addEventListener('click', () => toggleViewOption('roomLabels'));
    }

    if (toggleObjectLabelsBtn) {
        toggleObjectLabelsBtn.addEventListener('click', () => toggleViewOption('objectLabels'));
    }

    if (toggleDimensionsBtn) {
        toggleDimensionsBtn.addEventListener('click', () => toggleViewOption('dimensions'));
    }

    // View mode buttons
    if (standardViewBtn) {
        standardViewBtn.innerHTML = `
            <span class="material-icons">home</span>
            Standard
        `;
        standardViewBtn.addEventListener('click', () => setViewMode('standard'));
    }

    if (measurementViewBtn) {
        measurementViewBtn.innerHTML = `
            <span class="material-icons">architecture</span>
            Measurement
        `;
        measurementViewBtn.addEventListener('click', () => setViewMode('measurement'));
    }

    if (furnitureViewBtn) {
        furnitureViewBtn.innerHTML = `
            <span class="material-icons">chair</span>
            Furniture
        `;
        furnitureViewBtn.addEventListener('click', () => setViewMode('furniture'));
    }

    // --- Core Functions ---

    function initializeUI() {
        updateFloorIndicator();
    }

    function loadBlueprint() {
        showLoading(true);
        console.log("Fetching latest blueprint from API...");

        fetch('/api/blueprint/latest')
            .then(response => {
                console.log("API response status:", response.status);
                if (!response.ok) {
                    // Enhanced error handling - get more details from the response
                    if (response.status === 404) {
                        throw new Error('No blueprint found. Please generate one first.');
                    }
                    return response.json().then(err => {
                        throw new Error(err.error || `Failed to load blueprint: ${response.statusText}`);
                    }).catch(e => {
                        throw new Error(`Failed to load blueprint: ${response.statusText}`);
                    });
                }
                return response.json();
            })
            .then(data => {
                if (!data || !data.rooms || data.rooms.length === 0) {
                    throw new Error('Blueprint data is empty or invalid. Please generate a new one.');
                }

                blueprint = data;
                console.log("Blueprint loaded successfully:", blueprint);
                showLoading(false);
                updateBlueprintDisplay();
                showStatus('Blueprint loaded successfully', 'success');
            })
            .catch(error => {
                console.error('Error loading blueprint:', error);
                showLoading(false);

                // Show a more prominent error message in the canvas area
                showNoBlueprintMessage(error.message || 'Could not load blueprint. Please generate one.');

                // Also show in status area
                showStatus(error.message || 'Could not load blueprint. Please generate one.', 'error', 10000);
            });
    }

    function showNoBlueprintMessage(message) {
        // Create or update no-blueprint message in the canvas container
        let noBlueprintMsg = document.getElementById('no-blueprint-message');
        if (!noBlueprintMsg) {
            noBlueprintMsg = document.createElement('div');
            noBlueprintMsg.id = 'no-blueprint-message';
            noBlueprintMsg.style.position = 'absolute';
            noBlueprintMsg.style.top = '50%';
            noBlueprintMsg.style.left = '50%';
            noBlueprintMsg.style.transform = 'translate(-50%, -50%)';
            noBlueprintMsg.style.textAlign = 'center';
            noBlueprintMsg.style.padding = '20px';
            noBlueprintMsg.style.backgroundColor = 'rgba(0,0,0,0.7)';
            noBlueprintMsg.style.color = 'white';
            noBlueprintMsg.style.borderRadius = '8px';
            noBlueprintMsg.style.maxWidth = '80%';
            noBlueprintMsg.style.zIndex = '100';
            canvasContainer.appendChild(noBlueprintMsg);
        }

        noBlueprintMsg.innerHTML = `
            <h3>No Blueprint Available</h3>
            <p>${message}</p>
            <button id="generate-now-btn" class="primary-btn" style="margin-top: 15px">
                <span class="material-icons">autorenew</span>
                Generate Blueprint Now
            </button>
        `;

        // Add click handler for the generate button
        document.getElementById('generate-now-btn').addEventListener('click', generateBlueprint);
    }

    function generateBlueprint() {
        showLoading(true);
        showStatus('Generating blueprint...', 'info');

        // Remove any no-blueprint message if present
        const noBlueprintMsg = document.getElementById('no-blueprint-message');
        if (noBlueprintMsg) {
            noBlueprintMsg.remove();
        }

        fetch('/api/blueprint/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({})  // Add any generation parameters here if needed
        })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => {
                        throw new Error(err.error || `Failed to start generation: ${response.statusText}`);
                    }).catch(e => {
                        throw new Error(`Failed to start generation: ${response.statusText}`);
                    });
                }
                return response.json();
            })
            .then(data => {
                showStatus('Blueprint generation started. This may take a few minutes...', 'info', 15000);
                // Poll for completion or wait for a reasonable time then reload
                setTimeout(() => {
                    loadBlueprint();
                }, 15000);  // Wait 15 seconds then try to load the new blueprint
            })
            .catch(error => {
                console.error('Error generating blueprint:', error);
                showLoading(false);
                showStatus(`Failed to generate blueprint: ${error.message}`, 'error', 10000);
            });
    }

    function updateBlueprintDisplay() {
        if (!blueprint || !blueprint.rooms) {
            console.warn('No blueprint data available');
            showNoBlueprintMessage('No blueprint data available. Please generate one.');
            return;
        }

        // Hide any no-blueprint message if present
        const noBlueprintMsg = document.getElementById('no-blueprint-message');
        if (noBlueprintMsg) {
            noBlueprintMsg.remove();
        }

        // Filter rooms for the current floor
        const currentFloorRooms = blueprint.rooms.filter(room => {
            // Assuming rooms have a floor property
            return room.floor === currentFloor;
        });

        console.log(`Displaying ${currentFloorRooms.length} rooms on floor ${currentFloor}`);

        // Here you would update the canvas to display the rooms
        // For now, just log the number of rooms
        if (currentFloorRooms.length === 0) {
            showStatus(`No rooms found on floor ${currentFloor}`, 'info');
        } else {
            // Basic canvas rendering - draw simple boxes for rooms
            const canvas = document.getElementById('blueprint-canvas');
            if (canvas) {
                const ctx = canvas.getContext('2d');
                if (ctx) {
                    // Resize canvas to fit container
                    canvas.width = canvas.parentElement.clientWidth;
                    canvas.height = canvas.parentElement.clientHeight;

                    // Clear canvas
                    ctx.clearRect(0, 0, canvas.width, canvas.height);

                    // Draw rooms
                    const scale = 50;  // pixels per meter
                    const offsetX = canvas.width / 2;
                    const offsetY = canvas.height / 2;

                    ctx.fillStyle = '#e3f2fd';
                    ctx.strokeStyle = '#1976d2';
                    ctx.lineWidth = 2;

                    currentFloorRooms.forEach(room => {
                        if (room.shape && room.shape.points && room.shape.points.length > 2) {
                            ctx.beginPath();
                            ctx.moveTo(
                                offsetX + room.shape.points[0].x * scale,
                                offsetY + room.shape.points[0].y * scale
                            );

                            for (let i = 1; i < room.shape.points.length; i++) {
                                ctx.lineTo(
                                    offsetX + room.shape.points[i].x * scale,
                                    offsetY + room.shape.points[i].y * scale
                                );
                            }

                            ctx.closePath();
                            ctx.fill();
                            ctx.stroke();

                            // Add room name
                            if (room.name) {
                                // Calculate center of the room
                                let centerX = 0, centerY = 0;
                                room.shape.points.forEach(point => {
                                    centerX += point.x;
                                    centerY += point.y;
                                });
                                centerX /= room.shape.points.length;
                                centerY /= room.shape.points.length;

                                ctx.fillStyle = '#000';
                                ctx.font = '14px Arial';
                                ctx.textAlign = 'center';
                                ctx.fillText(
                                    room.name,
                                    offsetX + centerX * scale,
                                    offsetY + centerY * scale
                                );

                                // Reset fill style
                                ctx.fillStyle = '#e3f2fd';
                            }
                        }
                    });
                }
            }
        }
    }

    function changeFloor(direction) {
        // Calculate new floor
        const newFloor = currentFloor + direction;

        // Update if floor exists in blueprint
        if (blueprint && blueprint.rooms) {
            const floorExists = blueprint.rooms.some(room => room.floor === newFloor);

            if (floorExists) {
                currentFloor = newFloor;
                updateFloorIndicator();
                updateBlueprintDisplay();
                return;
            }

            // If floor doesn't exist in blueprint but direction is valid, still allow navigation
            // This assumes we might not have all floors populated yet
            if ((direction > 0 && newFloor <= 10) || (direction < 0 && newFloor >= -3)) {
                currentFloor = newFloor;
                updateFloorIndicator();
                updateBlueprintDisplay();
                showStatus(`No rooms on floor ${currentFloor} yet`, 'info');
            } else {
                showStatus(`Cannot navigate to floor ${newFloor}`, 'warning');
            }
        } else {
            // No blueprint loaded, just navigate within reasonable limits
            if ((direction > 0 && newFloor <= 10) || (direction < 0 && newFloor >= -3)) {
                currentFloor = newFloor;
                updateFloorIndicator();
                showStatus(`Floor changed to ${currentFloor}, but no blueprint is loaded`, 'info');
            } else {
                showStatus(`Cannot navigate beyond floor limits`, 'warning');
            }
        }
    }

    function updateFloorIndicator() {
        if (currentFloorIndicator) {
            if (currentFloor === 0) {
                currentFloorIndicator.textContent = 'Ground Floor';
            } else if (currentFloor > 0) {
                currentFloorIndicator.textContent = `Floor ${currentFloor}`;
            } else {
                currentFloorIndicator.textContent = `Basement ${Math.abs(currentFloor)}`;
            }
        }
    }

    function toggleUnitSystem() {
        unitSystem = unitSystem === 'metric' ? 'imperial' : 'metric';

        // Update button text
        if (toggleUnitSystemBtn) {
            toggleUnitSystemBtn.innerHTML = `
                <span class="material-icons">straighten</span>
                ${unitSystem === 'metric' ? 'Metric' : 'Imperial'}
            `;
        }

        // Update the display with new unit system
        updateBlueprintDisplay();
        showStatus(`Unit system changed to ${unitSystem}`, 'info');
    }

    function toggleViewOption(option) {
        // Implementation for toggling view options like room labels, object labels, etc.
        const buttonMap = {
            'roomLabels': toggleRoomLabelsBtn,
            'objectLabels': toggleObjectLabelsBtn,
            'dimensions': toggleDimensionsBtn
        };

        const button = buttonMap[option];

        if (button) {
            // Toggle the active class
            button.classList.toggle('active');

            // Update the blueprint display based on new settings
            updateBlueprintDisplay();
        }
    }

    function setViewMode(mode) {
        currentView = mode;

        // Update active class on view buttons
        if (standardViewBtn) standardViewBtn.classList.toggle('active', mode === 'standard');
        if (measurementViewBtn) measurementViewBtn.classList.toggle('active', mode === 'measurement');
        if (furnitureViewBtn) furnitureViewBtn.classList.toggle('active', mode === 'furniture');

        // Update the blueprint display based on the new view mode
        updateBlueprintDisplay();
        showStatus(`View changed to ${mode} mode`, 'info');
    }

    // --- UI Helper Functions ---

    function showLoading(isLoading) {
        if (loadingSpinner) {
            loadingSpinner.style.display = isLoading ? 'flex' : 'none';
        }
    }

    function showStatus(message, type = 'info', duration = 5000) {
        if (statusMessage) {
            statusMessage.textContent = message;
            statusMessage.className = 'status-message';
            statusMessage.classList.add(type);
            statusMessage.style.display = 'block';

            // Make error messages more prominent
            if (type === 'error') {
                statusMessage.style.backgroundColor = '#ffebee';
                statusMessage.style.color = '#c62828';
                statusMessage.style.padding = '15px';
                statusMessage.style.border = '1px solid #c62828';
                statusMessage.style.borderRadius = '4px';
                statusMessage.style.margin = '10px 0';
            }

            // Hide after a delay
            setTimeout(() => {
                statusMessage.style.display = 'none';
            }, duration);
        }
    }

    // --- Canvas Drawing Functions ---

    // These would be implemented to draw the actual blueprint
    // They would render rooms, walls, furniture, etc. based on the current floor,
    // view mode, and other display options
});
