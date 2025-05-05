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

        fetch('/api/blueprint/latest')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to load blueprint');
                }
                return response.json();
            })
            .then(data => {
                blueprint = data;
                showLoading(false);
                updateBlueprintDisplay();
                showStatus('Blueprint loaded successfully', 'success');
            })
            .catch(error => {
                console.error('Error loading blueprint:', error);
                showLoading(false);
                showStatus('Could not load blueprint. Please generate one.', 'error');
            });
    }

    function generateBlueprint() {
        showLoading(true);
        showStatus('Generating blueprint...', 'info');

        fetch('/api/blueprint/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({})  // Add any generation parameters here if needed
        })
            .then(response => response.json())
            .then(data => {
                showStatus('Blueprint generation started. This may take a few minutes...', 'info');
                // Poll for completion or wait for a reasonable time then reload
                setTimeout(() => {
                    loadBlueprint();
                }, 10000);  // Wait 10 seconds then try to load the new blueprint
            })
            .catch(error => {
                console.error('Error generating blueprint:', error);
                showLoading(false);
                showStatus('Failed to generate blueprint', 'error');
            });
    }

    function updateBlueprintDisplay() {
        if (!blueprint || !blueprint.rooms) {
            console.warn('No blueprint data available');
            return;
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
        }

        // TODO: Implement actual blueprint rendering on canvas
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

    function showStatus(message, type = 'info') {
        if (statusMessage) {
            statusMessage.textContent = message;
            statusMessage.className = 'status-message';
            statusMessage.classList.add(type);
            statusMessage.style.display = 'block';

            // Hide after a delay
            setTimeout(() => {
                statusMessage.style.display = 'none';
            }, 5000);
        }
    }

    // --- Canvas Drawing Functions ---

    // These would be implemented to draw the actual blueprint
    // They would render rooms, walls, furniture, etc. based on the current floor,
    // view mode, and other display options
});
