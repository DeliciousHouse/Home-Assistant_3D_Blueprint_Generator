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
let showRoomLabels = true; // Show room labels by default
let showObjectLabels = false; // Don't show object labels by default to prevent clutter
let showDimensions = false; // Option to display room dimensions
let selectedRoom = null; // Track selected room for highlighting
let hoveredObject = null; // Track hovered object for tooltips
let viewMode = 'standard'; // standard, measurement, or furniture
let unitSystem = 'metric'; // Default to metric ('metric' or 'imperial')

// Unit conversion utilities
const METERS_TO_FEET = 3.28084;
const SQUARE_METERS_TO_SQUARE_FEET = 10.7639;

function metersToFeet(meters) {
    return meters * METERS_TO_FEET;
}

function squareMetersToSquareFeet(squareMeters) {
    return squareMeters * SQUARE_METERS_TO_SQUARE_FEET;
}

function formatDistance(meters) {
    if (unitSystem === 'metric') {
        return `${meters.toFixed(1)}m`;
    } else {
        const feet = metersToFeet(meters);
        // For values less than 1 foot, show in inches
        if (feet < 1) {
            return `${(feet * 12).toFixed(1)}"`;
        }
        // Format as feet and inches for more precision
        const wholeFeet = Math.floor(feet);
        const inches = Math.round((feet - wholeFeet) * 12);
        if (inches === 0) {
            return `${wholeFeet}'`;
        } else {
            return `${wholeFeet}'${inches}"`;
        }
    }
}

function formatArea(squareMeters) {
    if (unitSystem === 'metric') {
        return `${squareMeters.toFixed(1)}m²`;
    } else {
        const squareFeet = squareMetersToSquareFeet(squareMeters);
        return `${squareFeet.toFixed(1)}ft²`;
    }
}

// Constants
const COLORS = {
    ROOMS: ['#4285F4', '#EA4335', '#FBBC05', '#34A853', '#FF6D01', '#46BDC6', '#9C27B0', '#2196F3'],
    WALLS: '#555555',
    SELECTED: '#FF9900',
    DOORS: '#8B4513',
    WINDOWS: '#87CEFA',
    TEXT: '#000000',
    OBJECTS: {
        'sofa': '#7b68ee',
        'coffee_table': '#8b4513',
        'tv_stand': '#2f4f4f',
        'bookshelf': '#deb887',
        'armchair': '#6a5acd',
        'refrigerator': '#b0c4de',
        'stove': '#696969',
        'sink': '#b0e0e6',
        'kitchen_cabinet': '#a9a9a9',
        'kitchen_island': '#a9a9a9',
        'bed': '#6495ed',
        'wardrobe': '#daa520',
        'nightstand': '#d2b48c',
        'dresser': '#d2b48c',
        'desk': '#8b4513',
        'office_chair': '#2f4f4f',
        'filing_cabinet': '#a9a9a9',
        'computer': '#808080',
        'dining_table': '#8b4513',
        'dining_chair': '#a0522d',
        'toilet': '#f0f8ff',
        'shower': '#87ceeb',
        'bathtub': '#b0e0e6',
        'mirror': '#c0c0c0',
        'default': '#9370db'
    },
    GRID: '#DDDDDD',
    GRID_MAIN: '#AAAAAA',
    BACKGROUND: '#f8f9fa',
    MEASUREMENT: '#FF4500'
};

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
        generateBtn.addEventListener('click', function() {
            // Show loading spinner
            document.getElementById('loading-spinner').style.display = 'flex';

            // Add status message
            showStatus('Generating blueprint... This may take a minute.');

            // Make API call to generate a new blueprint
            fetch('/api/blueprint/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'  // Ensure this header is set correctly
                },
                body: JSON.stringify({})  // Empty JSON object as body
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                // Hide loading spinner
                document.getElementById('loading-spinner').style.display = 'none';

                // Handle success
                showStatus('Blueprint generated successfully!', 'success');

                // Load the new blueprint
                loadBlueprint();
            })
            .catch(error => {
                // Hide loading spinner
                document.getElementById('loading-spinner').style.display = 'none';

                // Show error message
                console.error('Error generating blueprint:', error);
                showStatus('Error generating blueprint. Please try again.', 'error');
            });
        });
    }

    // Floor navigation
    const floorUp = document.getElementById('floor-up');
    const floorDown = document.getElementById('floor-down');
    if (floorUp) {
        floorUp.addEventListener('click', () => changeFloor(1));
    }
    if (floorDown) {
        floorDown.addEventListener('click', () => changeFloor(-1));
    }

    // View options
    const roomLabelsToggle = document.getElementById('toggle-room-labels');
    if (roomLabelsToggle) {
        roomLabelsToggle.addEventListener('click', () => {
            showRoomLabels = !showRoomLabels;
            roomLabelsToggle.classList.toggle('active', showRoomLabels);
            renderBlueprint(blueprint);
        });
        // Set initial state
        roomLabelsToggle.classList.toggle('active', showRoomLabels);
    }

    const objectLabelsToggle = document.getElementById('toggle-object-labels');
    if (objectLabelsToggle) {
        objectLabelsToggle.addEventListener('click', () => {
            showObjectLabels = !showObjectLabels;
            objectLabelsToggle.classList.toggle('active', showObjectLabels);
            renderBlueprint(blueprint);
        });
    }

    const dimensionsToggle = document.getElementById('toggle-dimensions');
    if (dimensionsToggle) {
        dimensionsToggle.addEventListener('click', () => {
            showDimensions = !showDimensions;
            dimensionsToggle.classList.toggle('active', showDimensions);
            renderBlueprint(blueprint);
        });
    }

    // Unit system toggle
    const unitSystemToggle = document.getElementById('toggle-unit-system');
    if (unitSystemToggle) {
        unitSystemToggle.addEventListener('click', () => {
            // Toggle between 'metric' and 'imperial'
            unitSystem = unitSystem === 'metric' ? 'imperial' : 'metric';

            // Update button text
            const unitDisplay = document.getElementById('unit-display');
            if (unitDisplay) {
                unitDisplay.textContent = unitSystem.charAt(0).toUpperCase() + unitSystem.slice(1);
            }

            // Update UI
            unitSystemToggle.classList.toggle('active', unitSystem === 'imperial');

            // Re-render with new units
            renderBlueprint(blueprint);

            // If a room is selected, update its details panel
            if (selectedRoom) {
                showRoomDetails(selectedRoom);
            }
        });
    }

    // View mode selection
    const standardViewBtn = document.getElementById('standard-view');
    const measurementViewBtn = document.getElementById('measurement-view');
    const furnitureViewBtn = document.getElementById('furniture-view');

    if (standardViewBtn) {
        standardViewBtn.addEventListener('click', () => {
            viewMode = 'standard';
            setActiveViewButton(standardViewBtn);
            renderBlueprint(blueprint);
        });
    }

    if (measurementViewBtn) {
        measurementViewBtn.addEventListener('click', () => {
            viewMode = 'measurement';
            setActiveViewButton(measurementViewBtn);
            renderBlueprint(blueprint);
        });
    }

    if (furnitureViewBtn) {
        furnitureViewBtn.addEventListener('click', () => {
            viewMode = 'furniture';
            setActiveViewButton(furnitureViewBtn);
            renderBlueprint(blueprint);
        });
    }

    // Canvas interactions
    if (canvas) {
        canvas.addEventListener('wheel', handleZoom, { passive: false });
        canvas.addEventListener('mousedown', startDrag);
        canvas.addEventListener('mousemove', drag);
        canvas.addEventListener('mouseup', endDrag);
        canvas.addEventListener('mouseleave', endDrag);

        // For room selection and object info
        canvas.addEventListener('click', handleCanvasClick);
        canvas.addEventListener('mousemove', handleCanvasHover);
    }
}

function setActiveViewButton(activeButton) {
    // Remove active class from all view buttons
    document.querySelectorAll('.view-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    // Add active class to the selected button
    activeButton.classList.add('active');
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
    ctx.fillStyle = COLORS.BACKGROUND;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw grid
    drawGrid();

    // Show message if no blueprint
    if (!blueprint) {
        ctx.fillStyle = '#888';
        ctx.font = 'bold 14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('No blueprint data available.', canvas.width / 2, canvas.height / 2 - 20);
        ctx.font = '12px Arial';
        ctx.fillText('Generate a blueprint using the button above.', canvas.width / 2, canvas.height / 2 + 10);
    }
}

function drawGrid() {
    // Draw light grid
    ctx.strokeStyle = COLORS.GRID;
    ctx.lineWidth = 0.5;

    // Calculate grid size based on scale
    const gridSize = 50 * camera.scale;

    // Calculate grid offset based on camera position
    const offsetX = (camera.x % gridSize);
    const offsetY = (camera.y % gridSize);

    // Draw vertical grid lines
    for (let x = offsetX; x < canvas.width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
    }

    // Draw horizontal grid lines
    for (let y = offsetY; y < canvas.height; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
    }

    // Draw main grid lines
    ctx.strokeStyle = COLORS.GRID_MAIN;
    ctx.lineWidth = 1;

    // Calculate main grid size (every 5 normal grid lines)
    const mainGridSize = gridSize * 5;
    const mainOffsetX = (camera.x % mainGridSize);
    const mainOffsetY = (camera.y % mainGridSize);

    // Draw main vertical grid lines
    for (let x = mainOffsetX; x < canvas.width; x += mainGridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
    }

    // Draw main horizontal grid lines
    for (let y = mainOffsetY; y < canvas.height; y += mainGridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
    }
}

// API Functions
function fetchBlueprint() {
    showLoading('Loading blueprint...');

    fetch('/api/blueprint/latest')
        .then(response => {
            if (!response.ok) {
                throw new Error(`Failed to fetch blueprint: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            blueprint = data;
            hideLoading();
            updateScene(data);
        })
        .catch(error => {
            hideLoading();
            displayErrorMessage(`Error loading blueprint: ${error.message}`);
            // Continue rendering empty canvas without data
            renderEmptyCanvas();
        });
}

function renderBlueprint(blueprintData) {
    // Clear canvas and draw grid
    renderEmptyCanvas();

    if (!blueprintData || !blueprintData.rooms) {
        return;
    }

    // Filter rooms by current floor
    const roomsToRender = blueprintData.rooms.filter(room => {
        return room.floor === currentFloor;
    });

    // First pass: Draw all room backgrounds
    roomsToRender.forEach((room, index) => {
        drawRoomBackground(room, index);
    });

    // Draw walls
    if (blueprintData.walls) {
        const wallsToRender = blueprintData.walls.filter(wall => {
            // Here we could filter walls by floor, but for simplicity we're showing all walls
            // In a real implementation, walls would have floor info
            return true;
        });

        wallsToRender.forEach(wall => {
            drawWall(wall);
        });
    }

    // Draw doors and windows if available
    if (blueprintData.doors) {
        blueprintData.doors.filter(door => door.floor === currentFloor).forEach(door => {
            drawDoor(door);
        });
    }

    if (blueprintData.windows) {
        blueprintData.windows.filter(window => window.floor === currentFloor).forEach(window => {
            drawWindow(window);
        });
    }

    // Draw objects (furniture) if in furniture view or standard view
    if ((viewMode === 'furniture' || viewMode === 'standard') && blueprintData.objects) {
        const objectsToRender = blueprintData.objects.filter(obj => {
            // Find the room for this object
            const room = blueprintData.rooms.find(r => r.id === obj.room_id);
            return room && room.floor === currentFloor;
        });

        objectsToRender.forEach(object => {
            drawObject(object);
        });
    }

    // Draw room labels and dimensions in a second pass (so they appear on top)
    if (showRoomLabels || showDimensions || viewMode === 'measurement') {
        roomsToRender.forEach(room => {
            if (showRoomLabels) {
                drawRoomLabel(room);
            }

            if (showDimensions || viewMode === 'measurement') {
                drawRoomDimensions(room);
            }
        });
    }

    // Draw measurement indicators if in measurement mode
    if (viewMode === 'measurement') {
        drawMeasurementGrid();
    }

    // Update floor indicator
    updateFloorIndicator();
}

function updateScene(blueprintData) {
    if (!blueprintData || !blueprintData.rooms || blueprintData.rooms.length === 0) {
        displayErrorMessage('Blueprint data is empty or invalid.');
        renderEmptyCanvas();
        return;
    }

    // Calculate content bounds
    calculateContentBounds(blueprintData);

    // Auto-scale on first load if needed
    if (autoScale) {
        applyAutoScale();
        autoScale = false; // Only auto-scale once
    }

    // Set initial floor if not already set
    if (blueprint && blueprint.rooms && blueprint.rooms.length > 0) {
        // Find all available floors
        const floors = [...new Set(blueprint.rooms.map(room => room.floor))].sort((a, b) => a - b);

        if (!floors.includes(currentFloor)) {
            currentFloor = floors[0] || 0; // Set to first floor if current floor not available
        }
    }

    // Render blueprint
    renderBlueprint(blueprintData);
}

// Calculate the bounds of all content for auto-scaling
function calculateContentBounds(blueprintData) {
    contentBounds = {
        minX: Infinity,
        minY: Infinity,
        maxX: -Infinity,
        maxY: -Infinity
    };

    // Include rooms in bounds
    if (blueprintData.rooms) {
        blueprintData.rooms.forEach(room => {
            if (room.bounds) {
                contentBounds.minX = Math.min(contentBounds.minX, room.bounds.min.x);
                contentBounds.minY = Math.min(contentBounds.minY, room.bounds.min.y);
                contentBounds.maxX = Math.max(contentBounds.maxX, room.bounds.max.x);
                contentBounds.maxY = Math.max(contentBounds.maxY, room.bounds.max.y);
            }
        });
    }

    // Include walls in bounds
    if (blueprintData.walls) {
        blueprintData.walls.forEach(wall => {
            contentBounds.minX = Math.min(contentBounds.minX, wall.start.x, wall.end.x);
            contentBounds.minY = Math.min(contentBounds.minY, wall.start.y, wall.end.y);
            contentBounds.maxX = Math.max(contentBounds.maxX, wall.start.x, wall.end.x);
            contentBounds.maxY = Math.max(contentBounds.maxY, wall.start.y, wall.end.y);
        });
    }

    // If no bounds were found, set defaults
    if (contentBounds.minX === Infinity) {
        contentBounds = { minX: -5, minY: -5, maxX: 5, maxY: 5 };
    }
}

// Apply auto-scaling to fit the blueprint in the canvas
function applyAutoScale() {
    if (!canvas) return;

    // Calculate content dimensions
    const contentWidth = contentBounds.maxX - contentBounds.minX;
    const contentHeight = contentBounds.maxY - contentBounds.minY;

    // Calculate content center
    const contentCenterX = (contentBounds.minX + contentBounds.maxX) / 2;
    const contentCenterY = (contentBounds.minY + contentBounds.maxY) / 2;

    // Calculate scale factor to fit content in canvas (with padding)
    const padding = 50; // Pixels of padding around the content
    const scaleX = (canvas.width - padding * 2) / contentWidth;
    const scaleY = (canvas.height - padding * 2) / contentHeight;

    // Use the smaller scale factor to ensure all content fits
    camera.scale = Math.min(scaleX, scaleY, 5); // Limit scale to avoid extreme zoom

    // Center the content
    camera.x = canvas.width / 2 - contentCenterX * camera.scale;
    camera.y = canvas.height / 2 - contentCenterY * camera.scale;
}

function drawRoomBackground(room, index) {
    if (!room.bounds) return;

    // Use the room bounds to draw a polygon
    const minX = worldToScreenX(room.bounds.min.x);
    const minY = worldToScreenY(room.bounds.min.y);
    const maxX = worldToScreenX(room.bounds.max.x);
    const maxY = worldToScreenY(room.bounds.max.y);

    // Select color - alternate colors for adjacent rooms
    ctx.fillStyle = room === selectedRoom ?
        COLORS.SELECTED :
        COLORS.ROOMS[index % COLORS.ROOMS.length];

    // Set opacity for better visibility
    ctx.globalAlpha = 0.5;

    // Draw room rectangle
    ctx.beginPath();
    ctx.rect(minX, minY, maxX - minX, maxY - minY);
    ctx.fill();

    // Reset opacity
    ctx.globalAlpha = 1.0;
}

function drawWall(wall) {
    if (!wall.start || !wall.end) return;

    // Convert world coordinates to screen coordinates
    const startX = worldToScreenX(wall.start.x);
    const startY = worldToScreenY(wall.start.y);
    const endX = worldToScreenX(wall.end.x);
    const endY = worldToScreenY(wall.end.y);

    // Calculate wall thickness in screen coordinates
    const thickness = wall.thickness ? wall.thickness * camera.scale : 2;

    ctx.lineWidth = thickness;
    ctx.strokeStyle = COLORS.WALLS;

    // Draw the wall line
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(endX, endY);
    ctx.stroke();
}

function drawDoor(door) {
    if (!door.start || !door.end) return;

    // Convert world coordinates to screen coordinates
    const startX = worldToScreenX(door.start.x);
    const startY = worldToScreenY(door.start.y);
    const endX = worldToScreenX(door.end.x);
    const endY = worldToScreenY(door.end.y);

    // Calculate thickness
    const thickness = door.thickness ? door.thickness * camera.scale : 1;

    ctx.lineWidth = thickness;
    ctx.strokeStyle = COLORS.DOORS;

    // Draw the door as a line with arc for door swing
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(endX, endY);
    ctx.stroke();

    // Draw door swing arc if direction is specified
    if (door.swing_direction) {
        const radius = Math.sqrt((endX - startX) ** 2 + (endY - startY) ** 2);
        const angle = Math.atan2(endY - startY, endX - startX);
        const swingAngle = door.swing_direction === 'left' ? Math.PI / 2 : -Math.PI / 2;

        ctx.beginPath();
        ctx.arc(startX, startY, radius, angle, angle + swingAngle, door.swing_direction === 'right');
        ctx.stroke();
    }
}

function drawWindow(window) {
    if (!window.start || !window.end) return;

    // Convert world coordinates to screen coordinates
    const startX = worldToScreenX(window.start.x);
    const startY = worldToScreenY(window.start.y);
    const endX = worldToScreenX(window.end.x);
    const endY = worldToScreenY(window.end.y);

    // Draw the window as a dashed line
    ctx.lineWidth = 1;
    ctx.strokeStyle = COLORS.WINDOWS;
    ctx.setLineDash([5, 2]);

    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(endX, endY);
    ctx.stroke();

    // Reset line dash
    ctx.setLineDash([]);

    // Draw window sill
    const dx = endX - startX;
    const dy = endY - startY;
    const length = Math.sqrt(dx * dx + dy * dy);
    const perpX = -dy / length * 5; // Perpendicular vector scaled
    const perpY = dx / length * 5;

    ctx.beginPath();
    ctx.moveTo(startX + perpX, startY + perpY);
    ctx.lineTo(endX + perpX, endY + perpY);
    ctx.stroke();
}

function drawObject(object) {
    if (!object.position || !object.dimensions) return;

    const x = worldToScreenX(object.position.x);
    const y = worldToScreenY(object.position.y);

    // Scale dimensions to screen coordinates
    const width = object.dimensions.width * camera.scale;
    const depth = object.dimensions.depth * camera.scale;

    // Determine if this object is being hovered
    const isHovered = hoveredObject === object;

    // Set color based on object type
    ctx.fillStyle = isHovered ?
        COLORS.SELECTED :
        COLORS.OBJECTS[object.type] || COLORS.OBJECTS.default;

    // Apply rotation if specified
    ctx.save();
    ctx.translate(x, y);
    if (object.rotation) {
        ctx.rotate(object.rotation * Math.PI / 180);
    }

    // Draw object shape
    ctx.beginPath();

    // Different shapes for different furniture types
    switch (object.type) {
        case 'toilet':
            ctx.ellipse(0, 0, width/2, depth/2, 0, 0, Math.PI * 2);
            break;
        case 'sink':
        case 'shower':
            ctx.ellipse(0, 0, width/2, depth/2, 0, 0, Math.PI * 2);
            break;
        case 'bathtub':
            // Rounded rectangle
            const cornerRadius = Math.min(width, depth) / 4;
            drawRoundedRect(-width/2, -depth/2, width, depth, cornerRadius);
            break;
        default:
            // Default rectangle
            ctx.rect(-width/2, -depth/2, width, depth);
            break;
    }

    ctx.fill();

    // Add outline for better visibility
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Draw object label if enabled or hovered
    if (showObjectLabels || isHovered) {
        ctx.fillStyle = '#000000';
        ctx.font = '10px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        // Format the label
        let label = object.type.replace(/_/g, ' ');
        label = label.charAt(0).toUpperCase() + label.slice(1); // Capitalize

        ctx.fillText(label, 0, 0);
    }

    ctx.restore();
}

// Helper function for drawing rounded rectangles
function drawRoundedRect(x, y, width, height, radius) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.arcTo(x + width, y, x + width, y + radius, radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.arcTo(x + width, y + height, x + width - radius, y + height, radius);
    ctx.lineTo(x + radius, y + height);
    ctx.arcTo(x, y + height, x, y + height - radius, radius);
    ctx.lineTo(x, y + radius);
    ctx.arcTo(x, y, x + radius, y, radius);
    ctx.closePath();
}

function drawRoomLabel(room) {
    if (!room.bounds) return;

    // Calculate center position of the room
    const centerX = worldToScreenX((room.bounds.min.x + room.bounds.max.x) / 2);
    const centerY = worldToScreenY((room.bounds.min.y + room.bounds.max.y) / 2);

    // Determine room name/label
    let roomLabel = 'Unknown';
    if (room.name) {
        roomLabel = room.name;
    } else if (room.area_id) {
        // Prettify area_id
        roomLabel = room.area_id.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    } else if (room.type) {
        roomLabel = room.type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    // Draw background for better readability
    ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
    ctx.font = 'bold 12px Arial';
    const labelWidth = ctx.measureText(roomLabel).width + 6;
    const labelHeight = 16;
    ctx.fillRect(centerX - labelWidth/2, centerY - labelHeight/2, labelWidth, labelHeight);

    // Draw text
    ctx.fillStyle = '#000000';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(roomLabel, centerX, centerY);
}

function drawRoomDimensions(room) {
    if (!room.bounds || !room.dimensions) return;

    const minX = worldToScreenX(room.bounds.min.x);
    const minY = worldToScreenY(room.bounds.min.y);
    const maxX = worldToScreenX(room.bounds.max.x);
    const maxY = worldToScreenY(room.bounds.max.y);

    // Width dimension
    const width = formatDistance(room.dimensions.width);
    drawDimensionLine(minX, minY - 15, maxX, minY - 15, width);

    // Length dimension
    const length = formatDistance(room.dimensions.length);
    drawDimensionLine(maxX + 15, minY, maxX + 15, maxY, length);

    // Area dimension if available
    if (room.dimensions.area) {
        const area = formatArea(room.dimensions.area);
        ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
        ctx.font = '10px Arial';
        const areaTextWidth = ctx.measureText(area).width + 6;
        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2 + 15;
        ctx.fillRect(centerX - areaTextWidth/2, centerY - 8, areaTextWidth, 16);

        ctx.fillStyle = COLORS.MEASUREMENT;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(area, centerX, centerY);
    }
}

function drawDimensionLine(x1, y1, x2, y2, label) {
    // Draw dimension line
    ctx.strokeStyle = COLORS.MEASUREMENT;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();

    // Draw arrows
    const angle = Math.atan2(y2 - y1, x2 - x1);
    drawArrowhead(x1, y1, angle);
    drawArrowhead(x2, y2, angle + Math.PI);

    // Draw dimension label
    const centerX = (x1 + x2) / 2;
    const centerY = (y1 + y2) / 2;

    ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
    ctx.font = '10px Arial';
    const labelWidth = ctx.measureText(label).width + 6;
    ctx.fillRect(centerX - labelWidth/2, centerY - 8, labelWidth, 16);

    ctx.fillStyle = COLORS.MEASUREMENT;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, centerX, centerY);
}

function drawArrowhead(x, y, angle) {
    const arrowLength = 10;
    const arrowWidth = 5;

    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(
        x - arrowLength * Math.cos(angle - Math.PI/6),
        y - arrowLength * Math.sin(angle - Math.PI/6)
    );
    ctx.lineTo(
        x - arrowLength * Math.cos(angle + Math.PI/6),
        y - arrowLength * Math.sin(angle + Math.PI/6)
    );
    ctx.closePath();
    ctx.fillStyle = COLORS.MEASUREMENT;
    ctx.fill();
}

function drawMeasurementGrid() {
    // Draw a more detailed grid with measurements
    const gridSize = 1; // 1 meter grid
    const gridSizeScreen = gridSize * camera.scale;

    // Calculate grid offset based on camera position
    const offsetX = (camera.x % gridSizeScreen);
    const offsetY = (camera.y % gridSizeScreen);

    // Calculate world coordinates of the first visible grid line
    const firstVisibleX = screenToWorldX(0) - (screenToWorldX(0) % gridSize);
    const firstVisibleY = screenToWorldY(0) - (screenToWorldY(0) % gridSize);

    ctx.strokeStyle = 'rgba(255, 0, 0, 0.3)';
    ctx.lineWidth = 0.5;
    ctx.font = '8px Arial';
    ctx.fillStyle = 'rgba(255, 0, 0, 0.7)';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';

    // Draw vertical grid lines with measurements
    for (let x = offsetX; x < canvas.width; x += gridSizeScreen) {
        const worldX = screenToWorldX(x);

        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();

        // Draw measurement label
        ctx.fillText(formatDistance(worldX), x + 2, 2);
    }

    // Draw horizontal grid lines with measurements
    for (let y = offsetY; y < canvas.height; y += gridSizeScreen) {
        const worldY = screenToWorldY(y);

        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();

        // Draw measurement label
        ctx.fillText(formatDistance(worldY), 2, y + 2);
    }
}

function updateFloorIndicator() {
    const floorIndicator = document.getElementById('current-floor');
    if (floorIndicator) {
        floorIndicator.textContent = `Floor ${currentFloor}`;
    }
}

function changeFloor(delta) {
    if (!blueprint || !blueprint.rooms) return;

    // Find all available floors
    const floors = [...new Set(blueprint.rooms.map(room => room.floor))].sort((a, b) => a - b);

    if (floors.length === 0) return;

    // Find current floor index
    const currentIndex = floors.indexOf(currentFloor);
    if (currentIndex === -1) {
        // Current floor not found, reset to first floor
        currentFloor = floors[0];
    } else {
        // Calculate new index
        const newIndex = Math.max(0, Math.min(floors.length - 1, currentIndex + delta));
        currentFloor = floors[newIndex];
    }

    // Update floor indicator and render blueprint
    updateFloorIndicator();
    renderBlueprint(blueprint);
}

// Coordinate conversion functions
function worldToScreenX(worldX) {
    return worldX * camera.scale + camera.x;
}

function worldToScreenY(worldY) {
    return worldY * camera.scale + camera.y;
}

function screenToWorldX(screenX) {
    return (screenX - camera.x) / camera.scale;
}

function screenToWorldY(screenY) {
    return (screenY - camera.y) / camera.scale;
}

// Action functions
function generateBlueprint() {
    showLoading('Generating blueprint...');

    fetch('/api/blueprint/generate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({})
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Failed to start generation: ${response.statusText}`);
        }
        return response.json();
    })
    .then(data => {
        displayStatusMessage(data.message || 'Blueprint generation started.');
        // Start polling for blueprint
        pollBlueprintStatus();
    })
    .catch(error => {
        hideLoading();
        displayErrorMessage(`Error starting generation: ${error.message}`);
    });
}

function pollBlueprintStatus() {
    let attempts = 0;
    const maxAttempts = 30; // Maximum number of polling attempts
    const pollInterval = 2000; // Poll every 2 seconds

    function poll() {
        attempts++;

        fetch('/api/blueprint/latest')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Failed to fetch blueprint: ${response.statusText}`);
                }
                return response.json();
            })
            .then(data => {
                // Check if this is a new blueprint (compare timestamps)
                const newTimestamp = data.generated_at;
                const oldTimestamp = blueprint ? blueprint.generated_at : null;

                if (newTimestamp !== oldTimestamp) {
                    // New blueprint available
                    blueprint = data;
                    hideLoading();
                    displayStatusMessage('Blueprint generation complete!');
                    autoScale = true; // Auto-scale for the new blueprint
                    updateScene(data);
                    return;
                }

                // Continue polling if max attempts not reached
                if (attempts < maxAttempts) {
                    setTimeout(poll, pollInterval);
                } else {
                    hideLoading();
                    displayStatusMessage('Blueprint generation taking longer than expected. Please check back later.');
                }
            })
            .catch(error => {
                hideLoading();
                displayErrorMessage(`Error checking blueprint status: ${error.message}`);
            });
    }

    // Start polling
    setTimeout(poll, pollInterval);
}

// Canvas interaction handlers
function handleZoom(e) {
    e.preventDefault();

    // Get mouse position in screen coordinates
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    // Get mouse position in world coordinates before zoom
    const worldX = screenToWorldX(mouseX);
    const worldY = screenToWorldY(mouseY);

    // Apply zoom factor based on scroll direction
    if (e.deltaY < 0) {
        // Zoom in
        camera.scale *= zoomFactor;
    } else {
        // Zoom out
        camera.scale /= zoomFactor;
    }

    // Limit zoom level
    camera.scale = Math.min(Math.max(camera.scale, 0.1), 10);

    // Adjust camera position to zoom toward mouse position
    camera.x = mouseX - worldX * camera.scale;
    camera.y = mouseY - worldY * camera.scale;

    // Re-render the blueprint
    renderBlueprint(blueprint);
}

function startDrag(e) {
    if (e.button !== 0) return; // Only handle left mouse button

    isDragging = true;

    const rect = canvas.getBoundingClientRect();
    lastMousePos = {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
    };
}

function drag(e) {
    if (!isDragging) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    // Calculate the movement delta
    const deltaX = mouseX - lastMousePos.x;
    const deltaY = mouseY - lastMousePos.y;

    // Update camera position
    camera.x += deltaX;
    camera.y += deltaY;

    // Update last mouse position
    lastMousePos = {
        x: mouseX,
        y: mouseY
    };

    // Re-render the blueprint
    renderBlueprint(blueprint);
}

function endDrag() {
    isDragging = false;
}

function handleCanvasClick(e) {
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    // Convert to world coordinates
    const worldX = screenToWorldX(mouseX);
    const worldY = screenToWorldY(mouseY);

    // Check for room selection
    if (blueprint && blueprint.rooms) {
        const previousSelection = selectedRoom;
        selectedRoom = null;

        blueprint.rooms.filter(room => room.floor === currentFloor).forEach(room => {
            // Simple bounding box check
            if (room.bounds &&
                worldX >= room.bounds.min.x && worldX <= room.bounds.max.x &&
                worldY >= room.bounds.min.y && worldY <= room.bounds.max.y) {
                selectedRoom = room;
            }
        });

        // Re-render only if selection changed
        if (selectedRoom !== previousSelection) {
            renderBlueprint(blueprint);

            // Show room details if a room is selected
            if (selectedRoom) {
                showRoomDetails(selectedRoom);
            } else {
                hideRoomDetails();
            }
        }
    }

    // If in measurement mode, display coordinates
    if (viewMode === 'measurement') {
        displayStatusMessage(`Position: (${formatDistance(worldX)}, ${formatDistance(worldY)})`);
    }
}

function handleCanvasHover(e) {
    if (isDragging) return; // Skip hover detection during drag

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    // Convert to world coordinates
    const worldX = screenToWorldX(mouseX);
    const worldY = screenToWorldY(mouseY);

    // Check for object hover
    const previousHover = hoveredObject;
    hoveredObject = null;

    if (blueprint && blueprint.objects) {
        blueprint.objects.filter(obj => {
            // Find the room for this object
            const room = blueprint.rooms.find(r => r.id === obj.room_id);
            return room && room.floor === currentFloor;
        }).forEach(obj => {
            // Simple distance check for object hover
            const dx = obj.position.x - worldX;
            const dy = obj.position.y - worldY;
            const distance = Math.sqrt(dx*dx + dy*dy);
            const hoverRadius = (obj.dimensions.width + obj.dimensions.depth) / 4;

            if (distance <= hoverRadius) {
                hoveredObject = obj;
            }
        });
    }

    // Re-render only if hover changed
    if (hoveredObject !== previousHover) {
        renderBlueprint(blueprint);
    }

    // Update cursor
    canvas.style.cursor = (hoveredObject || selectedRoom) ? 'pointer' : 'default';
}

function showRoomDetails(room) {
    const detailsPanel = document.getElementById('room-details');
    if (!detailsPanel) return;

    // Format room information
    let roomName = room.name || (room.area_id ? room.area_id.replace(/_/g, ' ') : 'Room');
    roomName = roomName.charAt(0).toUpperCase() + roomName.slice(1); // Capitalize

    const width = room.dimensions ? formatDistance(room.dimensions.width) : 'N/A';
    const length = room.dimensions ? formatDistance(room.dimensions.length) : 'N/A';
    const area = room.dimensions ? formatArea(room.dimensions.area) : 'N/A';

    // Populate details panel
    detailsPanel.innerHTML = `
        <h3>${roomName}</h3>
        <p><strong>Dimensions:</strong> ${width} × ${length}</p>
        <p><strong>Area:</strong> ${area}</p>
        <p><strong>Floor:</strong> ${room.floor || 0}</p>
    `;

    detailsPanel.style.display = 'block';
}

function hideRoomDetails() {
    const detailsPanel = document.getElementById('room-details');
    if (detailsPanel) {
        detailsPanel.style.display = 'none';
    }
}

// UI helpers
function showLoading(message) {
    if (loadingSpinner) {
        loadingSpinner.style.display = 'flex';
        const messageElement = loadingSpinner.querySelector('.message');
        if (messageElement) {
            messageElement.textContent = message;
        }
    }
}

function hideLoading() {
    if (loadingSpinner) {
        loadingSpinner.style.display = 'none';
    }
}

function displayStatusMessage(message, duration = 3000) {
    if (statusMessage) {
        statusMessage.textContent = message;
        statusMessage.className = 'status-message';
        statusMessage.style.display = 'block';

        setTimeout(() => {
            statusMessage.style.display = 'none';
        }, duration);
    }
}

function displayErrorMessage(message, duration = 5000) {
    if (statusMessage) {
        statusMessage.textContent = message;
        statusMessage.className = 'status-message error';
        statusMessage.style.display = 'block';

        setTimeout(() => {
            statusMessage.style.display = 'none';
        }, duration);
    }
}
