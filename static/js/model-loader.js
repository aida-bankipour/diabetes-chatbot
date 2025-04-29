// Advanced model loader for handling fallbacks when 3D model is unavailable

let anatomyScene, anatomyCamera, anatomyRenderer, anatomyControls;
let anatomyModel;
let isAnatomyPanelActive = false;
let is3DModelLoaded = false;
const originalMaterials = new Map();
let tooltip;

// Initialize the 3D or 2D anatomy viewer based on model availability
function initAnatomyViewer() {
    // Create tooltip element
    tooltip = document.createElement('div');
    tooltip.className = 'tooltip';
    document.body.appendChild(tooltip);

    // Get the container
    const container = document.getElementById('anatomy-canvas');

    // Try to load 3D model first
    try3DModel(container);
}

// Attempt to load the 3D model
function try3DModel(container) {
    // First, clear any existing content in the container
    // Remove the loading indicator placeholder
    const loadingElement = container.querySelector('.flex.items-center.justify-center.h-full');
    if (loadingElement) {
        container.removeChild(loadingElement);
    }

    // Set up the 3D scene
    anatomyScene = new THREE.Scene();
    anatomyScene.background = new THREE.Color(0xf0f0f0);

    // Set up the camera
    anatomyCamera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    anatomyCamera.position.z = 3;
    anatomyCamera.position.y = 1;

    // Set up the renderer with proper size and pixel ratio
    anatomyRenderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true // Add alpha channel for better integration
    });
    anatomyRenderer.setSize(container.clientWidth, container.clientHeight);
    anatomyRenderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(anatomyRenderer.domElement);

    // Set up lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7); // Increased brightness
    anatomyScene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0); // Increased brightness
    directionalLight.position.set(1, 1, 1);
    anatomyScene.add(directionalLight);

    // Add a second directional light from another angle for better illumination
    const secondaryLight = new THREE.DirectionalLight(0xffffff, 0.5);
    secondaryLight.position.set(-1, 0.5, -1);
    anatomyScene.add(secondaryLight);

    // Set up controls with better damping and limits
    anatomyControls = new THREE.OrbitControls(anatomyCamera, anatomyRenderer.domElement);
    anatomyControls.enableDamping = true;
    anatomyControls.dampingFactor = 0.1; // Smoother controls
    anatomyControls.screenSpacePanning = true; // Better panning
    anatomyControls.minDistance = 1.5;
    anatomyControls.maxDistance = 10;
    anatomyControls.autoRotate = false; // Don't auto-rotate initially

    // Show temporary loading message
    const tempLoadingDiv = document.createElement('div');
    tempLoadingDiv.style.position = 'absolute';
    tempLoadingDiv.style.top = '50%';
    tempLoadingDiv.style.left = '50%';
    tempLoadingDiv.style.transform = 'translate(-50%, -50%)';
    tempLoadingDiv.style.color = '#6366f1';
    tempLoadingDiv.style.fontWeight = 'bold';
    tempLoadingDiv.textContent = 'در حال بارگذاری...';
    container.appendChild(tempLoadingDiv);

    // Try to load the 3D model with error handling
    const loader = new THREE.GLTFLoader();
    loader.load('/static/models/human_body.glb',
        // Success callback
        function (gltf) {
            // Remove temporary loading message
            if (container.contains(tempLoadingDiv)) {
                container.removeChild(tempLoadingDiv);
            }

            anatomyModel = gltf.scene;

            // Properly scale and center the model
            anatomyModel.scale.set(1.5, 1.5, 1.5); // Larger size for better visibility
            anatomyScene.add(anatomyModel);

            // Store original materials and set up interaction
            anatomyModel.traverse((node) => {
                if (node.isMesh) {
                    // Create a proper clone of the material to avoid shared references
                    const originalMaterial = node.material.clone();
                    originalMaterials.set(node.uuid, originalMaterial);

                    // Add better names for body parts
                    node.userData.name = node.name || "بخشی از بدن";

                    // Set up material for highlighting
                    node.material = node.material.clone();
                    node.material.emissive = new THREE.Color(0x000000);
                    node.material.emissiveIntensity = 0;
                }
            });

            // Properly fit camera to model
            const box = new THREE.Box3().setFromObject(anatomyModel);
            const size = box.getSize(new THREE.Vector3());
            const center = box.getCenter(new THREE.Vector3());

            // Position model at center of scene
            anatomyModel.position.x = -center.x;
            anatomyModel.position.y = -center.y + size.y / 3; // Adjust vertical position
            anatomyModel.position.z = -center.z;

            // Set camera to properly view the model
            anatomyCamera.position.set(0, size.y / 2, size.z * 2);
            anatomyControls.target.set(0, size.y / 3, 0);
            anatomyControls.update();

            // Add a slight rotation for better initial view
            anatomyModel.rotation.y = Math.PI / 12; // Slight rotation

            // Set up raycaster for mouse interactions
            setupRaycaster();

            is3DModelLoaded = true;

            // Start animation loop
            animate();
        },
        // Progress callback
        function (xhr) {
            // Update loading percentage if needed
            const percent = Math.floor((xhr.loaded / xhr.total) * 100);
            if (tempLoadingDiv) {
                tempLoadingDiv.textContent = `در حال بارگذاری... ${percent}%`;
            }
        },
        // Error callback
        function (error) {
            console.warn('Error loading the 3D model:', error);
            // Remove temporary loading message
            if (container.contains(tempLoadingDiv)) {
                container.removeChild(tempLoadingDiv);
            }
            // Fall back to SVG
            useSVGFallback(container);
        }
    );
}

// Use SVG fallback when 3D model fails to load
function useSVGFallback(container) {
    console.log("Switching to SVG fallback for anatomy visualization");

    // Clear container contents first
    while (container.firstChild) {
        container.removeChild(container.firstChild);
    }

    // Create SVG directly in the container (no iframe)
    const svgContainer = document.createElement('div');
    svgContainer.style.width = '100%';
    svgContainer.style.height = '100%';
    svgContainer.style.display = 'flex';
    svgContainer.style.justifyContent = 'center';
    svgContainer.style.alignItems = 'center';
    container.appendChild(svgContainer);

    // Load SVG content using fetch
    fetch('/static/models/human_body_fallback.svg')
        .then(response => response.text())
        .then(svgContent => {
            svgContainer.innerHTML = svgContent;

            // Get SVG document
            const svgElement = svgContainer.querySelector('svg');
            if (!svgElement) {
                console.error("SVG element not found in loaded content");
                return;
            }

            // Size SVG properly
            svgElement.setAttribute('width', '100%');
            svgElement.setAttribute('height', '100%');
            svgElement.style.maxHeight = '450px';

            // Find all body parts and add interactivity
            const bodyParts = svgContainer.querySelectorAll('.body-part');

            // Add event listeners to each body part
            bodyParts.forEach(part => {
                // Highlight on hover
                part.addEventListener('mouseover', function () {
                    this.style.cursor = 'pointer';
                    this.querySelectorAll('*[fill]:not([fill="none"])').forEach(el => {
                        el.setAttribute('data-original-fill', el.getAttribute('fill'));
                        el.setAttribute('fill', '#ff9999');
                    });

                    // Show tooltip
                    const partName = this.getAttribute('data-name');
                    tooltip.textContent = partName;
                    tooltip.style.display = 'block';

                    // Add a transition effect
                    this.style.transition = 'all 0.3s ease';
                    this.style.transform = 'scale(1.05)';
                });

                // Mouse move for tooltip
                part.addEventListener('mousemove', function (e) {
                    const rect = svgContainer.getBoundingClientRect();
                    tooltip.style.left = `${rect.left + e.offsetX + 10}px`;
                    tooltip.style.top = `${rect.top + e.offsetY - 20}px`;
                });

                // Reset on mouseout
                part.addEventListener('mouseout', function () {
                    this.querySelectorAll('*[data-original-fill]').forEach(el => {
                        el.setAttribute('fill', el.getAttribute('data-original-fill'));
                        el.removeAttribute('data-original-fill');
                    });
                    tooltip.style.display = 'none';
                    this.style.transform = 'scale(1)';
                });

                // Click to show info
                part.addEventListener('click', function () {
                    const partName = this.getAttribute('data-name');
                    const bodyPartInfo = getBodyPartInfo(partName);
                    if (bodyPartInfo) {
                        addMessage(bodyPartInfo, true);
                    }

                    // Highlight this part and reset others
                    bodyParts.forEach(otherPart => {
                        if (otherPart !== this) {
                            otherPart.querySelectorAll('*[fill]:not([fill="none"])').forEach(el => {
                                const originalFill = el.getAttribute('data-original-fill') || el.getAttribute('fill');
                                el.setAttribute('fill', originalFill);
                            });
                            otherPart.style.transform = 'scale(1)';
                        }
                    });

                    this.querySelectorAll('*[fill]:not([fill="none"])').forEach(el => {
                        el.setAttribute('fill', '#ff5555');
                    });
                    this.style.transform = 'scale(1.1)';
                });
            });

            // Add a success message
            console.log("SVG fallback loaded successfully");
        })
        .catch(error => {
            console.error("Error loading SVG fallback:", error);
            svgContainer.innerHTML = '<div style="text-align:center;color:#ff3333">خطا در بارگذاری مدل. لطفاً صفحه را بارگذاری مجدد کنید.</div>';
        });
}

// Animation loop for 3D model
function animate() {
    if (is3DModelLoaded) {
        requestAnimationFrame(animate);
        anatomyControls.update();
        anatomyRenderer.render(anatomyScene, anatomyCamera);
    }
}

// Set up raycaster for 3D model interactions
function setupRaycaster() {
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    let hoveredObject = null;

    // Mouse move event for hover
    document.getElementById('anatomy-canvas').addEventListener('mousemove', (event) => {
        const canvas = document.getElementById('anatomy-canvas');
        const rect = canvas.getBoundingClientRect();

        // Calculate mouse position in normalized device coordinates (-1 to +1)
        mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        // Cast ray from the camera through the mouse position
        raycaster.setFromCamera(mouse, anatomyCamera);

        // Find intersections with interactive objects
        const intersects = raycaster.intersectObjects(anatomyScene.children, true);

        // Reset previous hover
        if (hoveredObject) {
            hoveredObject.material.emissiveIntensity = 0;
            tooltip.style.display = 'none';
        }

        // Set new hover
        if (intersects.length > 0) {
            const object = intersects[0].object;
            if (object.isMesh) {
                hoveredObject = object;
                hoveredObject.material.emissiveIntensity = 0.3;
                hoveredObject.material.emissive.set(0x007bff);

                // Show tooltip
                tooltip.textContent = object.userData.name;
                tooltip.style.display = 'block';
                tooltip.style.left = `${event.clientX}px`;
                tooltip.style.top = `${event.clientY - 28}px`;
            }
        } else {
            hoveredObject = null;
        }
    });

    // Mouse click event
    document.getElementById('anatomy-canvas').addEventListener('click', (event) => {
        const canvas = document.getElementById('anatomy-canvas');
        const rect = canvas.getBoundingClientRect();

        mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        raycaster.setFromCamera(mouse, anatomyCamera);
        const intersects = raycaster.intersectObjects(anatomyScene.children, true);

        if (intersects.length > 0) {
            const object = intersects[0].object;
            if (object.isMesh) {
                highlightBodyPart(object);

                // Map body parts to diabetes-related information
                const bodyPartInfo = getBodyPartInfo(object.name || "unknown");
                if (bodyPartInfo) {
                    // Add a message to the chat
                    addMessage(bodyPartInfo, true);
                }
            }
        }
    });
}

// Highlight body part in 3D model
function highlightBodyPart(object) {
    // Reset all materials
    anatomyModel.traverse((node) => {
        if (node.isMesh && node !== object) {
            const originalMaterial = originalMaterials.get(node.uuid);
            if (originalMaterial) {
                node.material = originalMaterial.clone();
                node.material.emissiveIntensity = 0;
            }
        }
    });

    // Highlight selected part
    object.material = object.material.clone();
    object.material.emissive.set(0xff0000);
    object.material.emissiveIntensity = 0.5;
}

// Get diabetes information for a body part
function getBodyPartInfo(bodyPart) {
    const bodyPartMapping = {
        "head": "سر و مغز: دیابت می‌تواند بر مغز تأثیر بگذارد و منجر به مشکلات شناختی و افزایش خطر سکته و آلزایمر شود.",
        "eye": "چشم: رتینوپاتی دیابتی یک عارضه شایع است که می‌تواند باعث تاری دید و حتی نابینایی شود.",
        "heart": "قلب: دیابت خطر بیماری‌های قلبی را افزایش می‌دهد، از جمله بیماری عروق کرونر و حمله قلبی.",
        "kidney": "کلیه: نفروپاتی دیابتی باعث آسیب به کلیه‌ها می‌شود و می‌تواند منجر به نارسایی کلیه شود.",
        "foot": "پا: نوروپاتی دیابتی می‌تواند به اعصاب پا آسیب برساند و منجر به بی‌حسی، زخم پای دیابتی و در موارد شدید قطع عضو شود.",
        "skin": "پوست: دیابت می‌تواند منجر به مشکلات پوستی مانند خشکی، خارش و عفونت‌های قارچی شود.",
        "leg": "پاها: دیابت می‌تواند جریان خون به پاها را کاهش دهد و باعث زخم، عفونت و در موارد شدید قطع عضو شود.",
        "arm": "دست‌ها: نوروپاتی دیابتی می‌تواند باعث بی‌حسی و سوزن سوزن شدن در دست‌ها شود.",
        "stomach": "معده و روده: دیابت می‌تواند بر دستگاه گوارش تأثیر بگذارد و باعث مشکلاتی مانند تأخیر در تخلیه معده شود.",
        "liver": "کبد: دیابت می‌تواند با بیماری کبد چرب غیرالکلی مرتبط باشد.",
        "pancreas": "پانکراس: این اندام تولیدکننده انسولین است و در دیابت نوع ۱، سیستم ایمنی بدن به سلول‌های بتا در پانکراس حمله می‌کند.",
        "muscle": "عضلات: دیابت می‌تواند بر توده عضلانی و قدرت تأثیر منفی بگذارد.",
        "brain": "مغز: دیابت می‌تواند خطر مشکلات شناختی، سکته مغزی و زوال عقل را افزایش دهد.",
        "hand": "دست‌ها: دیابت می‌تواند باعث سندرم تونل کارپال و سایر مشکلات عصبی در دست‌ها شود.",
        "torso": "بدن: دیابت یک بیماری سیستمیک است که می‌تواند بر چندین اندام بدن تأثیر بگذارد."
    };

    // Check if the bodyPart contains any of the keys (partial match)
    for (const [key, info] of Object.entries(bodyPartMapping)) {
        if (bodyPart.toLowerCase().includes(key)) {
            return info;
        }
    }

    return "اطلاعات خاصی در مورد تأثیر دیابت بر این قسمت از بدن در دسترس نیست.";
}

// Handle window resize
function onWindowResize() {
    const container = document.getElementById('anatomy-canvas');
    if (!container) return;

    if (is3DModelLoaded && anatomyRenderer) {
        const width = container.clientWidth;
        const height = container.clientHeight;

        anatomyCamera.aspect = width / height;
        anatomyCamera.updateProjectionMatrix();
        anatomyRenderer.setSize(width, height);
    }
}

// Toggle anatomy panel
function toggleAnatomyPanel() {
    const panel = document.getElementById('anatomy-panel');
    if (!panel) return;

    isAnatomyPanelActive = !isAnatomyPanelActive;
    panel.classList.toggle('active', isAnatomyPanelActive);

    if (isAnatomyPanelActive && !anatomyScene) {
        // Initialize viewer the first time panel is opened
        initAnatomyViewer();
    } else if (isAnatomyPanelActive) {
        // Resize when panel is reopened
        onWindowResize();
    }
}

// Reset camera view
function resetCamera() {
    if (!is3DModelLoaded) {
        // For SVG fallback, reload the iframe
        const container = document.getElementById('anatomy-canvas');
        const iframe = container.querySelector('iframe');
        if (iframe) {
            iframe.src = iframe.src;
        }
        return;
    }

    if (!anatomyControls || !anatomyCamera || !anatomyModel) return;

    // Reset camera position
    anatomyCamera.position.set(0, 1, 3);
    anatomyControls.target.set(0, 1, 0);
    anatomyControls.update();

    // Reset all materials to original
    anatomyModel.traverse((node) => {
        if (node.isMesh) {
            const originalMaterial = originalMaterials.get(node.uuid);
            if (originalMaterial) {
                node.material = originalMaterial.clone();
            }
        }
    });
}

// Add window resize event listener
window.addEventListener('resize', onWindowResize);
