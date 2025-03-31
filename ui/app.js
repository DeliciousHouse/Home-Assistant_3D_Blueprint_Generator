// Blueprint Generator Web UI JavaScript

// Main DOM elements
let canvas;
let ctx;
let statusMessage;
let loadingSpinner;

// Global state
let blueprint = null;
let currentFloor = 0;
let camera = { x: 0, y: 0, scale: 1 };
let isDragging = false;
let lastMousePos = { x: 0, y: 0 };

// Constants
const COLORS = [
    '#4285F4', '#EA4335', '#FBBC05', '#34A853',
    '#FF6D01', '#46BDC6', '#9C27B0', '#2196F3'
];

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    canvas = document.getElementById('blueprint-canvas');
    statusMessage = document.getElementById('status-message');
    loadingSpinner = document.getElementById('loading-spinner');

    // Initialize canvas
    if (canvas) {
        ctx = canvas.getContext('2d');
        initializeCanvas();
    }

    // Add event listeners
    setupEventListeners();

    // Initial data loading
    fetchBlueprint();
});

function setupEventListeners() {
    // Button event listeners
    const generateBtn = document.getElementById('generate-btn');
    if (generateBtn) {
        generateBtn.addEventListener('click', generateBlueprint);
    }

    // Floor navigation
    const floorUp = document.getElementById('floor-up');
    const floorDown = document.getElementById('floor-down');
    if (floorUp) floorUp.addEventListener('click', () => changeFloor(1));
    if (floorDown) floorDown.addEventListener('click', () => changeFloor(-1));

    // Canvas interactions
    if (canvas) {
        canvas.addEventListener('wheel', handleZoom);
        canvas.addEventListener('mousedown', startDrag);
        canvas.addEventListener('mousemove', drag);
        canvas.addEventListener('mouseup', endDrag);
        canvas.addEventListener('mouseout', endDrag);
    }
}

function initializeCanvas() {
    // Set canvas size to match container
    const container = canvas.parentElement;
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;

    // Initial render
    renderEmptyCanvas();
}

function renderEmptyCanvas() {
    // Clear canvas
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw grid
    drawGrid();

    // Show message if no blueprint
    if (!blueprint) {
        ctx.fillStyle = '#6c757d';
        ctx.font = '16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('No blueprint data available.', canvas.width / 2, canvas.height / 2);
        ctx.fillText('Click "Generate Blueprint" to create one.', canvas.width / 2, canvas.height / 2 + 30);
    }
}

function drawGrid() {
    // Draw light grid
    const gridSize = 50 * camera.scale;
    const offsetX = camera.x % gridSize;
    const offsetY = camera.y % gridSize;

    ctx.strokeStyle = '#e9ecef';
    ctx.lineWidth = 1;
    ctx.beginPath();

    // Vertical lines
    for (let x = offsetX; x < canvas.width; x += gridSize) {
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
    }

    // Horizontal lines
    for (let y = offsetY; y < canvas.height; y += gridSize) {
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
    }

    ctx.stroke();

    // Add scale indicator
    ctx.fillStyle = '#495057';
    ctx.font = '12px Arial';
    ctx.textAlign = 'left';
    ctx.fillText(`Scale: 1m = ${camera.scale.toFixed(1)}px`, 10, canvas.height - 10);
}

// API Functions
function fetchBlueprint() {
    showLoading('Loading blueprint...');

    fetch('/api/blueprint')
        .then(response => {
            if (response.status === 404) {
                // No blueprint available yet
                hideLoading();
                return null;
            }
            if (!response.ok) throw new Error('Network response was not ok');
            return response.json();
        })
        .then(data => {
            if (data) {
                blueprint = data;
                renderBlueprint(blueprint);
            }
            hideLoading();
        })
        .catch(error => {
            console.error('Error fetching blueprint:', error);
            displayErrorMessage('Failed to load blueprint. Check console for details.');
            hideLoading();
        });
}

function renderBlueprint(blueprint) {
    // This function should update the visual representation.
    // For now, let's just call updateScene which uses the global blueprint variable.
    console.log("Rendering blueprint data received from fetch...");
    // Assuming updateScene uses the global 'blueprint' variable which was set in fetchBlueprint
    updateScene(blueprint);
}

function updateScene(blueprint) {
    if (!ctx || !blueprint) return;

    // Clear canvas
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw grid
    drawGrid();

    // Get rooms for current floor
    const floorData = blueprint.floors.find(f => f.level === currentFloor);
    if (!floorData) {
        ctx.fillStyle = '#6c757d';
        ctx.font = '16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(`No data for floor ${currentFloor}`, canvas.width / 2, canvas.height / 2);
        return;
    }

    const floorRooms = blueprint.rooms.filter(r => floorData.rooms.includes(r.id));

    // Center the blueprint
    centerBlueprint(floorRooms);

    // Draw rooms
    floorRooms.forEach((room, index) => {
        drawRoom(room, index);
    });

    // Draw walls
    if (blueprint.walls) {
        blueprint.walls.forEach(wall => {
            // Only draw walls for current floor
            if (wall.floor === undefined || wall.floor === currentFloor) {
                drawWall(wall);
            }
        });
    }

    // Update floor indicator
    updateFloorIndicator();
}

function centerBlueprint(rooms) {
    if (!rooms || rooms.length === 0) return;

    // Find the bounds of all rooms
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;

    rooms.forEach(room => {
        const bounds = room.bounds;
        minX = Math.min(minX, bounds.min.x);
        maxX = Math.max(maxX, bounds.max.x);
        minY = Math.min(minY, bounds.min.y);
        maxY = Math.max(maxY, bounds.max.y);
    });

    // Calculate blueprint center and dimensions
    const blueprintWidth = maxX - minX;
    const blueprintHeight = maxY - minY;
    const blueprintCenterX = minX + blueprintWidth / 2;
    const blueprintCenterY = minY + blueprintHeight / 2;

    // Calculate canvas center
    const canvasCenterX = canvas.width / 2;
    const canvasCenterY = canvas.height / 2;

    // Adjust camera to center blueprint
    camera.x = canvasCenterX - blueprintCenterX * camera.scale;
    camera.y = canvasCenterY - blueprintCenterY * camera.scale;

    // Auto-scale to fit blueprint
    const scaleX = canvas.width / (blueprintWidth * 1.2);  // 1.2 for padding
    const scaleY = canvas.height / (blueprintHeight * 1.2);
    camera.scale = Math.min(scaleX, scaleY, 20);  // Limit max scale
}

function drawRoom(room, index) {
    const bounds = room.bounds;

    // Convert room coords to canvas coords
    const x1 = bounds.min.x * camera.scale + camera.x;
    const y1 = bounds.min.y * camera.scale + camera.y;
    const x2 = bounds.max.x * camera.scale + camera.x;
    const y2 = bounds.max.y * camera.scale + camera.y;

    // Draw room
    ctx.fillStyle = COLORS[index % COLORS.length] + '33';  // Add transparency
    ctx.strokeStyle = COLORS[index % COLORS.length];
    ctx.lineWidth = 2;

    ctx.beginPath();
    ctx.rect(x1, y1, x2 - x1, y2 - y1);
    ctx.fill();
    ctx.stroke();

    // Draw room name
    ctx.fillStyle = '#212529';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(room.name, (x1 + x2) / 2, (y1 + y2) / 2);
}

function drawWall(wall) {
    // Convert wall coords to canvas coords
    const x1 = wall.start.x * camera.scale + camera.x;
    const y1 = wall.start.y * camera.scale + camera.y;
    const x2 = wall.end.x * camera.scale + camera.x;
    const y2 = wall.end.y * camera.scale + camera.y;

    // Draw wall
    ctx.strokeStyle = '#212529';
    ctx.lineWidth = wall.thickness * camera.scale;

    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
}

function updateFloorIndicator() {
    const floorIndicator = document.getElementById('current-floor');
    if (floorIndicator) {
        floorIndicator.textContent = `Floor ${currentFloor}`;
    }
}

function changeFloor(delta) {
    if (!blueprint || !blueprint.floors) return;

    const floorLevels = blueprint.floors.map(f => f.level).sort((a, b) => a - b);
    const currentIndex = floorLevels.indexOf(currentFloor);

    if (currentIndex === -1) {
        // Current floor not found, default to first floor
        currentFloor = floorLevels[0];
    } else {
        const newIndex = Math.max(0, Math.min(floorLevels.length - 1, currentIndex + delta));
        currentFloor = floorLevels[newIndex];
    }

    renderBlueprint(blueprint);
}

// Action functions
function generateBlueprint() {
    showLoading('Generating blueprint...');

    // Make a POST request to generate the blueprint
    fetch('/api/blueprint/generate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
        .then(response => {
            if (!response.ok) throw new Error('Network response was not ok');
            return response.json();
        })
        .then(data => {
            if (data.status === 'success') {
                displayStatusMessage('Blueprint generation started. This may take a few moments...');

                // Poll for status until complete
                pollBlueprintStatus();
            } else {
                displayErrorMessage(data.message || 'Failed to start blueprint generation.');
                hideLoading();
            }
        })
        .catch(error => {
            console.error('Error generating blueprint:', error);
            displayErrorMessage('Failed to generate blueprint. Check console for details.');
            hideLoading();
        });
}

function pollBlueprintStatus() {
    fetch('/api/blueprint/status')
        .then(response => {
            if (!response.ok) throw new Error('Network response was not ok');
            return response.json();
        })
        .then(data => {
            const progress = Math.round(data.progress * 100);
            displayStatusMessage(`Blueprint generation: ${data.state} (${progress}%)...`);

            if (data.state === 'complete') {
                // Blueprint is ready, fetch it
                fetchBlueprint();
            } else {
                // Continue polling
                setTimeout(pollBlueprintStatus, 2000);
            }
        })
        .catch(error => {
            console.error('Error polling blueprint status:', error);
            displayErrorMessage('Failed to check blueprint status. Check console for details.');
            hideLoading();
        });
}

// Canvas interaction handlers
function handleZoom(e) {
    e.preventDefault();

    const mouseX = e.offsetX;
    const mouseY = e.offsetY;

    // Convert mouse position to world coordinates
    const worldX = (mouseX - camera.x) / camera.scale;
    const worldY = (mouseY - camera.y) / camera.scale;

    // Update scale
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    camera.scale = Math.max(0.1, Math.min(20, camera.scale * zoomFactor));

    // Adjust camera position to zoom into the mouse position
    camera.x = mouseX - worldX * camera.scale;
    camera.y = mouseY - worldY * camera.scale;

    // Redraw
    if (blueprint) {
        renderBlueprint(blueprint);
    } else {
        renderEmptyCanvas();
    }
}

function startDrag(e) {
    isDragging = true;
    lastMousePos = { x: e.offsetX, y: e.offsetY };
}

function drag(e) {
    if (!isDragging) return;

    const dx = e.offsetX - lastMousePos.x;
    const dy = e.offsetY - lastMousePos.y;

    camera.x += dx;
    camera.y += dy;

    lastMousePos = { x: e.offsetX, y: e.offsetY };

    // Redraw
    if (blueprint) {
        renderBlueprint(blueprint);
    } else {
        renderEmptyCanvas();
    }
}

function endDrag() {
    isDragging = false;
}

// UI helpers
function showLoading(message) {
    if (loadingSpinner) {
        loadingSpinner.classList.remove('d-none');
    }

    if (message && statusMessage) {
        statusMessage.textContent = message;
        statusMessage.classList.remove('d-none');
    }
}

function hideLoading() {
    if (loadingSpinner) {
        loadingSpinner.classList.add('d-none');
    }
}

function displayStatusMessage(message, duration = 3000) {
    if (!statusMessage) return;

    statusMessage.textContent = message;
    statusMessage.className = 'alert alert-info mt-3';
    statusMessage.classList.remove('d-none');

    setTimeout(() => {
        statusMessage.classList.add('d-none');
    }, duration);
}

function displayErrorMessage(message, duration = 5000) {
    // Simple implementation: update the status element with error styling
    console.error("Displaying Error:", message); // Log error to console too
    const statusEl = document.getElementById('status-message'); // Use the existing status element
    if (statusEl) {
        statusEl.textContent = `Error: ${message}`;
        statusEl.className = 'alert alert-danger mt-3';
        statusEl.classList.remove('d-none');

        setTimeout(() => {
            statusEl.classList.add('d-none');
        }, duration);
    }
}
