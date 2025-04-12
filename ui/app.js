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
let contentBounds = { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity };
let autoScale = true; // Flag to control initial auto-scaling
const zoomFactor = 1.1; // Define zoom factor constant

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
        canvas.style.cursor = 'grab'; // Set initial cursor
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
    console.log("Fetching blueprint from /api/blueprint..."); // Add log

    fetch('/api/blueprint')
        .then(response => {
            if (response.status === 404) {
                console.log("API returned 404 - No blueprint found."); // Add log
                hideLoading();
                blueprint = null; // Ensure blueprint is null if not found
                renderEmptyCanvas(); // Render the empty state
                return null;
            }
            if (!response.ok) {
                console.error(`Network response error: ${response.status} ${response.statusText}`); // Add log
                throw new Error(`Network response was not ok: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            if (data && data.success && data.blueprint) {
                console.log("Successfully fetched blueprint data:", data.blueprint); // Log fetched data
                blueprint = data.blueprint; // Assign ONLY the blueprint object
                renderBlueprint(blueprint); // Pass the correct object
            } else if (data && !data.success) {
                console.error("API call successful but returned error:", data.error);
                displayErrorMessage(data.error || "API returned an error.");
                blueprint = null;
                renderEmptyCanvas();
            } else {
                // Handle cases where data is null (e.g., from 404) or malformed
                console.log("No valid blueprint data received or data format incorrect.");
                blueprint = null;
                renderEmptyCanvas();
            }
            hideLoading();
        })
        .catch(error => {
            console.error('Error fetching blueprint:', error);
            displayErrorMessage('Failed to load blueprint. Check console for details.');
            blueprint = null; // Reset blueprint on error
            renderEmptyCanvas(); // Render empty state on error
            hideLoading();
        });
}

function renderBlueprint(blueprintData) {
    // This function should update the visual representation.
    console.log("Rendering blueprint data:", blueprintData); // Log data being passed

    if (!blueprintData) {
        console.error("No blueprint data to render");
        renderEmptyCanvas();
        return;
    }

    // Calculate bounds for auto-scaling
    calculateContentBounds(blueprintData);

    // Apply auto-scaling if enabled
    if (autoScale) {
        applyAutoScale();
        autoScale = false; // Only auto-scale once when blueprint first loads
    }

    // Update the scene with blueprint data
    updateScene(blueprintData);
}

function updateScene(blueprintData) {
    // Add checks at the beginning
    if (!ctx) {
        console.error("Canvas context (ctx) is not available.");
        return;
    }
    if (!blueprintData || typeof blueprintData !== 'object') {
        console.error("updateScene called with invalid blueprint data:", blueprintData);
        renderEmptyCanvas(); // Render empty state if data is bad
        return;
    }
    if (!blueprintData.floors || !blueprintData.rooms) {
        console.error("Blueprint data is missing 'floors' or 'rooms' array:", blueprintData);
        renderEmptyCanvas();
        return;
    }

    console.log(`Updating scene for floor ${currentFloor}`); // Log floor being rendered

    // Clear canvas
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw grid
    drawGrid();

    // Get rooms for current floor
    const floorData = blueprintData.floors.find(f => f.level === currentFloor);
    if (!floorData) {
        console.warn(`No floor data found for level ${currentFloor}`); // Changed to warning
        ctx.fillStyle = '#6c757d';
        ctx.font = '16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(`No data for floor ${currentFloor}`, canvas.width / 2, canvas.height / 2);
        return;
    }
    if (!floorData.rooms || !Array.isArray(floorData.rooms)) {
        console.error(`Floor data for level ${currentFloor} is missing 'rooms' array:`, floorData);
        renderEmptyCanvas();
        return;
    }

    const floorRooms = blueprintData.rooms.filter(r => floorData.rooms.includes(r.id));
    console.log(`Found ${floorRooms.length} rooms for floor ${currentFloor}`); // Log room count

    // Draw rooms
    floorRooms.forEach((room, index) => {
        console.log(`Drawing room: ${room.name} (ID: ${room.id})`); // Log room being drawn
        drawRoom(room, index);
    });

    // Draw walls
    if (blueprintData.walls && Array.isArray(blueprintData.walls)) {
        console.log(`Drawing ${blueprintData.walls.length} walls...`); // Log wall count
        blueprintData.walls.forEach(wall => {
            // Only draw walls for current floor (assuming walls have a floor property or are global)
            if (wall.floor === undefined || wall.floor === currentFloor) {
                console.log("Drawing wall:", wall); // Log wall being drawn
                drawWall(wall);
            }
        });
    } else {
        console.log("No walls array found in blueprint data or it's not an array.");
    }

    // Update floor indicator
    updateFloorIndicator();
}

// Calculate the bounds of all content for auto-scaling
function calculateContentBounds(blueprintData) {
    contentBounds = { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity };
    if (!blueprintData || !blueprintData.rooms) return;

    blueprintData.rooms.forEach(room => {
        if (room.bounds) {
            contentBounds.minX = Math.min(contentBounds.minX, room.bounds.min.x);
            contentBounds.minY = Math.min(contentBounds.minY, room.bounds.min.y);
            contentBounds.maxX = Math.max(contentBounds.maxX, room.bounds.max.x);
            contentBounds.maxY = Math.max(contentBounds.maxY, room.bounds.max.y);
        }
    });

    // Also consider walls if they exist
    if (blueprintData.walls) {
        blueprintData.walls.forEach(wall => {
            contentBounds.minX = Math.min(contentBounds.minX, wall.start.x, wall.end.x);
            contentBounds.minY = Math.min(contentBounds.minY, wall.start.y, wall.end.y);
            contentBounds.maxX = Math.max(contentBounds.maxX, wall.start.x, wall.end.x);
            contentBounds.maxY = Math.max(contentBounds.maxY, wall.start.y, wall.end.y);
        });
    }

    console.log("Calculated content bounds:", contentBounds);
}

// Apply auto-scaling to fit the blueprint in the canvas
function applyAutoScale() {
    if (contentBounds.minX === Infinity || !canvas) return; // No content or canvas not ready

    const contentWidth = contentBounds.maxX - contentBounds.minX;
    const contentHeight = contentBounds.maxY - contentBounds.minY;

    if (contentWidth <= 0 || contentHeight <= 0) {
        console.warn("Cannot auto-scale with zero or negative content dimensions.");
        // Reset to default scale/offset if content is invalid
        camera.scale = 10;
        camera.x = canvas.width / 2;
        camera.y = canvas.height / 2;
        return;
    }

    const padding = 50; // Pixels padding around the content
    const availableWidth = canvas.width - 2 * padding;
    const availableHeight = canvas.height - 2 * padding;

    // Calculate scale to fit content within available space
    const scaleX = availableWidth / contentWidth;
    const scaleY = availableHeight / contentHeight;
    camera.scale = Math.min(scaleX, scaleY); // Use the smaller scale to fit both dimensions

    // Calculate offsets to center the content
    const scaledContentWidth = contentWidth * camera.scale;
    const scaledContentHeight = contentHeight * camera.scale;
    camera.x = padding + (availableWidth - scaledContentWidth) / 2 - (contentBounds.minX * camera.scale);
    camera.y = padding + (availableHeight - scaledContentHeight) / 2 - (contentBounds.minY * camera.scale);

    console.log(`Auto-scaling applied: scale=${camera.scale.toFixed(2)}, offsetX=${camera.x.toFixed(2)}, offsetY=${camera.y.toFixed(2)}`);
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
    const delta = e.deltaY > 0 ? 1 / zoomFactor : zoomFactor;
    camera.scale = Math.max(0.1, Math.min(20, camera.scale * delta));

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
    canvas.style.cursor = 'grabbing';
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
    if (canvas) {
        canvas.style.cursor = 'grab';
    }
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
