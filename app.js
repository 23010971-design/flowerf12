// ============================================================
// YOLO11 REALTIME WEB APP
// iPhone Camera + ONNX Runtime Web
// Bounding Box + tô màu + tên + confidence
// ============================================================

const MODEL_PATH = "./best.onnx";
const INPUT_SIZE = 320;
const CONF_THRESHOLD = 0.60;
const IOU_THRESHOLD = 0.50;
const INFERENCE_INTERVAL = 100;


// ============================================================
// CLASS NAMES
// ============================================================

const CLASS_NAMES = [
    "CHIEU_TIM",              // ID 0
    "CHUOI_DUOC_DO_RUNG",     // ID 1
    "CHUOI_MO_KET",           // ID 2
    "DAM_BUT_CAM",            // ID 3
    "DAM_BUT_VANG",           // ID 4
    "HONG",                   // ID 5
    "HUYNH_ANH",              // ID 6
    "MAO_GA_DUOI_CONG",       // ID 7
    "MAU_DON_DO",             // ID 8
    "MAU_DON_TRANG",          // ID 9
    "MAU_DON_VANG",           // ID 10
    "MIA_DO",                 // ID 11
    "MO_TRANG",               // ID 12
    "NGOC_BUT"                // ID 13
];


// ============================================================
// CLASS COLORS
// ============================================================

const CLASS_COLORS = [
    [255, 255, 0],    // ID 0  - CHIEU_TIM
    [0, 200, 255],    // ID 1  - CHUOI_DUOC_DO_RUNG
    [255, 165, 0],    // ID 2  - CHUOI_MO_KET
    [0, 255, 0],      // ID 3  - DAM_BUT_CAM
    [200, 80, 255],   // ID 4  - DAM_BUT_VANG
    [255, 0, 0],      // ID 5  - HONG
    [0, 128, 255],    // ID 6  - HUYNH_ANH
    [255, 0, 255],    // ID 7  - MAO_GA_DUOI_CONG
    [128, 255, 0],    // ID 8  - MAU_DON_DO
    [255, 255, 255],  // ID 9  - MAU_DON_TRANG
    [0, 255, 255],    // ID 10 - MAU_DON_VANG
    [128, 0, 255],    // ID 11 - MIA_DO
    [255, 128, 0],    // ID 12 - MO_TRANG
    [0, 255, 128]    // ID 13 - NGOC_BUT
];


// ============================================================
// GLOBAL VARIABLES
// ============================================================

let session = null;
let stream = null;

let running = false;
let inferenceBusy = false;

let lastInferenceTime = 0;

let lastTime = performance.now();
let frameCounter = 0;
let fps = 0;


// ============================================================
// DOM
// ============================================================

const video =
    document.getElementById("video");

const overlay =
    document.getElementById("overlay");

const overlayCtx =
    overlay.getContext("2d");

const processingCanvas =
    document.getElementById("processingCanvas");

const processingCtx =
    processingCanvas.getContext(
        "2d",
        {
            willReadFrequently: true
        }
    );

const statusElement =
    document.getElementById("status");

const fpsElement =
    document.getElementById("fps");

const objectsElement =
    document.getElementById("objects");

const detectionsElement =
    document.getElementById("detections");

const loadButton =
    document.getElementById("loadButton");

const cameraButton =
    document.getElementById("cameraButton");

const stopButton =
    document.getElementById("stopButton");


// ============================================================
// STATUS
// ============================================================

function setStatus(text) {

    if (statusElement) {
        statusElement.textContent = text;
    }

    console.log("[YOLO]", text);
}


// ============================================================
// LOAD MODEL
// ============================================================

async function loadModel() {

    if (session) {

        setStatus(
            "YOLO model already loaded"
        );

        return true;
    }

    try {

        setStatus(
            "Loading YOLO model..."
        );

        if (loadButton) {
            loadButton.disabled = true;
        }


        // ----------------------------------------------------
        // TRY WEBGL
        // ----------------------------------------------------

        try {

            session =
                await ort.InferenceSession.create(
                    MODEL_PATH,
                    {
                        executionProviders: [
                            "webgl"
                        ],

                        graphOptimizationLevel:
                            "all"
                    }
                );

            setStatus(
                "YOLO loaded - WebGL"
            );

        } catch (webglError) {

            console.warn(
                "WebGL failed:",
                webglError
            );


            // ------------------------------------------------
            // FALLBACK WASM
            // ------------------------------------------------

            session =
                await ort.InferenceSession.create(
                    MODEL_PATH,
                    {
                        executionProviders: [
                            "wasm"
                        ],

                        graphOptimizationLevel:
                            "all"
                    }
                );

            setStatus(
                "YOLO loaded - WASM"
            );
        }


        console.log(
            "Input names:",
            session.inputNames
        );

        console.log(
            "Output names:",
            session.outputNames
        );

        console.log(
            "Input metadata:",
            session.inputMetadata
        );

        console.log(
            "Output metadata:",
            session.outputMetadata
        );


        return true;


    } catch (error) {

        console.error(
            "Model loading failed:",
            error
        );

        session = null;

        setStatus(
            "Model loading failed"
        );

        if (loadButton) {
            loadButton.disabled = false;
        }

        alert(
            "Cannot load best.onnx.\n\n" +
            "Check that best.onnx is in the same folder as index.html."
        );

        return false;
    }
}


// ============================================================
// CAMERA
// ============================================================

async function startCamera() {


    // --------------------------------------------------------
    // TỰ LOAD MODEL NẾU CHƯA LOAD
    // --------------------------------------------------------

    if (!session) {

        const loaded =
            await loadModel();

        if (!loaded) {
            return;
        }
    }


    try {


        // ----------------------------------------------------
        // STOP OLD CAMERA
        // ----------------------------------------------------

        if (stream) {

            stream
                .getTracks()
                .forEach(
                    track =>
                        track.stop()
                );

            stream = null;
        }


        setStatus(
            "Requesting camera..."
        );


        // ----------------------------------------------------
        // IPHONE REAR CAMERA
        // ----------------------------------------------------

        stream =
            await navigator.mediaDevices.getUserMedia(
                {

                    video: {

                        facingMode: {
                            ideal:
                                "environment"
                        },

                        width: {
                            ideal: 640
                        },

                        height: {
                            ideal: 480
                        },

                        frameRate: {
                            ideal: 30,
                            max: 30
                        }
                    },

                    audio: false
                }
            );


        video.srcObject =
            stream;


        await video.play();


        setupOverlay();


        running = true;

        inferenceBusy = false;

        lastInferenceTime = 0;


        setStatus(
            "Camera running"
        );


        requestAnimationFrame(
            inferenceLoop
        );


    } catch (error) {

        console.error(
            "Camera error:",
            error
        );

        setStatus(
            "Camera error"
        );

        alert(
            "Camera access failed.\n\n" +
            "Make sure you opened the website using HTTPS " +
            "and allowed Camera permission."
        );
    }
}


// ============================================================
// STOP CAMERA
// ============================================================

function stopCamera() {

    running = false;

    inferenceBusy = false;


    if (stream) {

        stream
            .getTracks()
            .forEach(
                track =>
                    track.stop()
            );

        stream = null;
    }


    video.srcObject = null;


    overlayCtx.clearRect(
        0,
        0,
        overlay.width,
        overlay.height
    );


    objectsElement.textContent =
        "0";


    detectionsElement.textContent =
        "No detection";


    fpsElement.textContent =
        "0";


    setStatus(
        session
            ? "Model loaded"
            : "Model not loaded"
    );
}


// ============================================================
// OVERLAY
// ============================================================

function setupOverlay() {

    if (
        !video.videoWidth ||
        !video.videoHeight
    ) {

        return;
    }


    overlay.width =
        video.videoWidth;

    overlay.height =
        video.videoHeight;
}


// ============================================================
// PREPROCESS
// ============================================================

function preprocess() {

    const videoWidth =
        video.videoWidth;

    const videoHeight =
        video.videoHeight;


    if (
        !videoWidth ||
        !videoHeight
    ) {

        return null;
    }


    processingCanvas.width =
        INPUT_SIZE;

    processingCanvas.height =
        INPUT_SIZE;


    processingCtx.clearRect(
        0,
        0,
        INPUT_SIZE,
        INPUT_SIZE
    );


    // --------------------------------------------------------
    // LETTERBOX
    // --------------------------------------------------------

    const scale =
        Math.min(
            INPUT_SIZE /
                videoWidth,

            INPUT_SIZE /
                videoHeight
        );


    const newWidth =
        videoWidth * scale;

    const newHeight =
        videoHeight * scale;


    const offsetX =
        (
            INPUT_SIZE -
            newWidth
        ) / 2;


    const offsetY =
        (
            INPUT_SIZE -
            newHeight
        ) / 2;


    // --------------------------------------------------------
    // BACKGROUND
    // --------------------------------------------------------

    processingCtx.fillStyle =
        "#808080";


    processingCtx.fillRect(
        0,
        0,
        INPUT_SIZE,
        INPUT_SIZE
    );


    // --------------------------------------------------------
    // DRAW CAMERA
    // --------------------------------------------------------

    processingCtx.drawImage(

        video,

        0,
        0,
        videoWidth,
        videoHeight,

        offsetX,
        offsetY,
        newWidth,
        newHeight
    );


    // --------------------------------------------------------
    // GET PIXELS
    // --------------------------------------------------------

    const imageData =
        processingCtx.getImageData(
            0,
            0,
            INPUT_SIZE,
            INPUT_SIZE
        );


    const pixels =
        imageData.data;


    const channelSize =
        INPUT_SIZE *
        INPUT_SIZE;


    const floatData =
        new Float32Array(
            channelSize * 3
        );


    // --------------------------------------------------------
    // RGB -> CHW
    // --------------------------------------------------------

    for (
        let i = 0;
        i < channelSize;
        i++
    ) {

        const p =
            i * 4;


        floatData[i] =
            pixels[p] / 255.0;


        floatData[
            channelSize + i
        ] =
            pixels[p + 1] / 255.0;


        floatData[
            channelSize * 2 + i
        ] =
            pixels[p + 2] / 255.0;
    }


    return {

        tensor:
            new ort.Tensor(
                "float32",
                floatData,
                [
                    1,
                    3,
                    INPUT_SIZE,
                    INPUT_SIZE
                ]
            ),

        scale:
            scale,

        offsetX:
            offsetX,

        offsetY:
            offsetY
    };
}


// ============================================================
// INFERENCE
// ============================================================

async function runInference() {

    const prep =
        preprocess();


    if (!prep) {
        return null;
    }


    const inputName =
        session.inputNames[0];


    const feeds = {};


    feeds[inputName] =
        prep.tensor;


    const results =
        await session.run(
            feeds
        );


    const outputName =
        session.outputNames[0];


    const output =
        results[outputName];


    return {

        output:
            output,

        scale:
            prep.scale,

        offsetX:
            prep.offsetX,

        offsetY:
            prep.offsetY
    };
}


// ============================================================
// POSTPROCESS
// YOLO OUTPUT [1,9,8400]
// ============================================================

function postprocess(
    output,
    scale,
    offsetX,
    offsetY
) {

    if (
        !output ||
        !output.dims ||
        output.dims.length !== 3
    ) {

        throw new Error(
            "YOLO output shape không hợp lệ"
        );
    }


    const data =
        output.data;


    const dims =
        output.dims;


    const numChannels =
        dims[1];


    const numBoxes =
        dims[2];


    const numClasses =
        numChannels - 4;


    const candidates = [];


    // --------------------------------------------------------
    // LOOP ALL BOXES
    // --------------------------------------------------------

    for (
        let i = 0;
        i < numBoxes;
        i++
    ) {


        const cx =
            data[i];


        const cy =
            data[
                numBoxes + i
            ];


        const w =
            data[
                2 * numBoxes + i
            ];


        const h =
            data[
                3 * numBoxes + i
            ];


        // ----------------------------------------------------
        // FIND BEST CLASS
        // ----------------------------------------------------

        let bestClass =
            -1;


        let bestScore =
            0;


        for (
            let c = 0;
            c < numClasses;
            c++
        ) {

            const score =
                data[
                    (4 + c) *
                    numBoxes +
                    i
                ];


            if (
                score >
                bestScore
            ) {

                bestScore =
                    score;

                bestClass =
                    c;
            }
        }


        // ----------------------------------------------------
        // CONFIDENCE
        // ----------------------------------------------------

        if (
            bestScore <
            CONF_THRESHOLD
        ) {

            continue;
        }


        // ----------------------------------------------------
        // XYWH -> XYXY
        // ----------------------------------------------------

        let x1 =
            cx - w / 2;


        let y1 =
            cy - h / 2;


        let x2 =
            cx + w / 2;


        let y2 =
            cy + h / 2;


        // ----------------------------------------------------
        // REMOVE LETTERBOX
        // ----------------------------------------------------

        x1 =
            (
                x1 -
                offsetX
            ) /
            scale;


        y1 =
            (
                y1 -
                offsetY
            ) /
            scale;


        x2 =
            (
                x2 -
                offsetX
            ) /
            scale;


        y2 =
            (
                y2 -
                offsetY
            ) /
            scale;


        // ----------------------------------------------------
        // CLAMP
        // ----------------------------------------------------

        x1 =
            Math.max(
                0,
                Math.min(
                    video.videoWidth,
                    x1
                )
            );


        y1 =
            Math.max(
                0,
                Math.min(
                    video.videoHeight,
                    y1
                )
            );


        x2 =
            Math.max(
                0,
                Math.min(
                    video.videoWidth,
                    x2
                )
            );


        y2 =
            Math.max(
                0,
                Math.min(
                    video.videoHeight,
                    y2
                )
            );


        if (
            x2 <= x1 ||
            y2 <= y1
        ) {

            continue;
        }


        candidates.push({

            x1:
                x1,

            y1:
                y1,

            x2:
                x2,

            y2:
                y2,

            score:
                bestScore,

            classId:
                bestClass
        });
    }


    return nms(
        candidates,
        IOU_THRESHOLD
    );
}


// ============================================================
// IOU
// ============================================================

function calculateIoU(
    a,
    b
) {

    const x1 =
        Math.max(
            a.x1,
            b.x1
        );


    const y1 =
        Math.max(
            a.y1,
            b.y1
        );


    const x2 =
        Math.min(
            a.x2,
            b.x2
        );


    const y2 =
        Math.min(
            a.y2,
            b.y2
        );


    const intersectionWidth =
        Math.max(
            0,
            x2 - x1
        );


    const intersectionHeight =
        Math.max(
            0,
            y2 - y1
        );


    const intersection =
        intersectionWidth *
        intersectionHeight;


    const areaA =
        Math.max(
            0,
            a.x2 - a.x1
        ) *
        Math.max(
            0,
            a.y2 - a.y1
        );


    const areaB =
        Math.max(
            0,
            b.x2 - b.x1
        ) *
        Math.max(
            0,
            b.y2 - b.y1
        );


    const union =
        areaA +
        areaB -
        intersection;


    if (
        union <= 0
    ) {

        return 0;
    }


    return (
        intersection /
        union
    );
}


// ============================================================
// NMS
// ============================================================

function nms(
    boxes,
    threshold
) {

    const sorted =
        [...boxes].sort(
            (a, b) =>
                b.score -
                a.score
        );


    const selected = [];


    while (
        sorted.length > 0
    ) {

        const current =
            sorted.shift();


        selected.push(
            current
        );


        for (
            let i =
                sorted.length - 1;
            i >= 0;
            i--
        ) {

            const box =
                sorted[i];


            // NMS riêng từng class
            if (
                box.classId !==
                current.classId
            ) {

                continue;
            }


            const iou =
                calculateIoU(
                    current,
                    box
                );


            if (
                iou >=
                threshold
            ) {

                sorted.splice(
                    i,
                    1
                );
            }
        }
    }


    return selected;
}


// ============================================================
// DRAW DETECTIONS
// TÔ MÀU + BOUNDING BOX + TÊN + CONFIDENCE
// ============================================================

function drawDetections(
    detections
) {

    overlayCtx.clearRect(
        0,
        0,
        overlay.width,
        overlay.height
    );


    if (
        !video.videoWidth ||
        !video.videoHeight
    ) {

        return;
    }


    const scaleX =
        overlay.width /
        video.videoWidth;


    const scaleY =
        overlay.height /
        video.videoHeight;


    detections.forEach(
        detection => {


            // ------------------------------------------------
            // BOX COORDINATES
            // ------------------------------------------------

            const x =
                detection.x1 *
                scaleX;


            const y =
                detection.y1 *
                scaleY;


            const width =
                (
                    detection.x2 -
                    detection.x1
                ) *
                scaleX;


            const height =
                (
                    detection.y2 -
                    detection.y1
                ) *
                scaleY;


            // ------------------------------------------------
            // CLASS COLOR
            // ------------------------------------------------

            const color =
                CLASS_COLORS[
                    detection.classId
                ] ||
                [0, 255, 0];


            const r =
                color[0];


            const g =
                color[1];


            const b =
                color[2];


            // ------------------------------------------------
            // CLASS NAME
            // ------------------------------------------------

            const className =
                CLASS_NAMES[
                    detection.classId
                ] ||
                `Class ${detection.classId}`;


            const confidence =
                (
                    detection.score *
                    100
                ).toFixed(1);


            const label =
                `${className} ${confidence}%`;


            // =================================================
            // 1. TÔ MÀU BÊN TRONG
            // =================================================

            overlayCtx.fillStyle =
                `rgba(
                    ${r},
                    ${g},
                    ${b},
                    0.25
                )`;


            overlayCtx.fillRect(
                x,
                y,
                width,
                height
            );


            // =================================================
            // 2. BOUNDING BOX
            // =================================================

            overlayCtx.strokeStyle =
                `rgb(
                    ${r},
                    ${g},
                    ${b}
                )`;


            overlayCtx.lineWidth =
                4;


            overlayCtx.strokeRect(
                x,
                y,
                width,
                height
            );


            // =================================================
            // 3. LABEL FONT
            // =================================================

            overlayCtx.font =
                "bold 16px Arial";


            const textWidth =
                overlayCtx.measureText(
                    label
                ).width;


            const labelWidth =
                textWidth + 16;


            const labelHeight =
                26;


            // =================================================
            // 4. LABEL POSITION
            // =================================================

            let labelX =
                x;


            let labelY =
                y -
                labelHeight;


            // Nếu box sát mép trên
            if (
                labelY < 0
            ) {

                labelY =
                    y;
            }


            // Không cho label vượt phải
            if (
                labelX +
                labelWidth >
                overlay.width
            ) {

                labelX =
                    overlay.width -
                    labelWidth;
            }


            // =================================================
            // 5. LABEL BACKGROUND
            // =================================================

            overlayCtx.fillStyle =
                `rgb(
                    ${r},
                    ${g},
                    ${b}
                )`;


            overlayCtx.fillRect(
                labelX,
                labelY,
                labelWidth,
                labelHeight
            );


            // =================================================
            // 6. LABEL TEXT
            // =================================================

            overlayCtx.fillStyle =
                "#ffffff";


            overlayCtx.textBaseline =
                "middle";


            overlayCtx.fillText(
                label,
                labelX + 8,
                labelY +
                labelHeight / 2
            );


            overlayCtx.textBaseline =
                "alphabetic";
        }
    );


    // ========================================================
    // OBJECT COUNT
    // ========================================================

    objectsElement.textContent =
        detections.length;


    // ========================================================
    // DETECTION LIST
    // ========================================================

    if (
        detections.length === 0
    ) {

        detectionsElement.textContent =
            "No detection";

        return;
    }


    detectionsElement.innerHTML =
        detections
            .map(
                detection => {

                    const className =
                        CLASS_NAMES[
                            detection.classId
                        ] ||
                        `Class ${detection.classId}`;


                    const confidence =
                        (
                            detection.score *
                            100
                        ).toFixed(1);


                    const color =
                        CLASS_COLORS[
                            detection.classId
                        ] ||
                        [0, 255, 0];


                    return `
                        <div style="
                            padding:6px 8px;
                            margin-bottom:5px;
                            border-radius:6px;
                            background:rgba(
                                ${color[0]},
                                ${color[1]},
                                ${color[2]},
                                0.15
                            );
                        ">
                            <strong>
                                ${className}
                            </strong>
                            - ${confidence}%
                        </div>
                    `;
                }
            )
            .join("");
}


// ============================================================
// FPS
// ============================================================

function updateFPS() {

    frameCounter++;


    const now =
        performance.now();


    const elapsed =
        now -
        lastTime;


    if (
        elapsed >= 1000
    ) {

        fps =
            frameCounter *
            1000 /
            elapsed;


        fpsElement.textContent =
            fps.toFixed(1);


        frameCounter =
            0;


        lastTime =
            now;
    }
}


// ============================================================
// REALTIME LOOP
// ============================================================

async function inferenceLoop(
    timestamp
) {

    if (!running) {
        return;
    }


    // Camera vẫn chạy liên tục.
    // YOLO chỉ inference mỗi INFERENCE_INTERVAL.
    if (
        video.readyState >= 2
    ) {


        if (
            timestamp -
            lastInferenceTime >=
            INFERENCE_INTERVAL
        ) {


            lastInferenceTime =
                timestamp;


            // Không cho inference chồng nhau
            if (
                !inferenceBusy
            ) {

                inferenceBusy =
                    true;


                try {

                    const result =
                        await runInference();


                    if (
                        result
                    ) {

                        const detections =
                            postprocess(
                                result.output,
                                result.scale,
                                result.offsetX,
                                result.offsetY
                            );


                        drawDetections(
                            detections
                        );


                        updateFPS();


                        if (
                            detections.length >
                            0
                        ) {

                            setStatus(
                                `Detected ${detections.length} object(s)`
                            );

                        } else {

                            setStatus(
                                "Scanning..."
                            );
                        }
                    }


                } catch (error) {

                    console.error(
                        "Inference error:",
                        error
                    );


                    setStatus(
                        "Inference error - see Console"
                    );


                } finally {

                    inferenceBusy =
                        false;
                }
            }
        }
    }


    requestAnimationFrame(
        inferenceLoop
    );
}


// ============================================================
// BUTTONS
// ============================================================

if (loadButton) {

    loadButton.addEventListener(
        "click",
        loadModel
    );
}


if (cameraButton) {

    cameraButton.addEventListener(
        "click",
        startCamera
    );
}


if (stopButton) {

    stopButton.addEventListener(
        "click",
        stopCamera
    );
}


// ============================================================
// VIDEO METADATA
// ============================================================

video.addEventListener(
    "loadedmetadata",
    () => {

        setupOverlay();
    }
);


// ============================================================
// WINDOW RESIZE
// ============================================================

window.addEventListener(
    "resize",
    () => {

        setupOverlay();
    }
);


// ============================================================
// INITIAL STATUS
// ============================================================

setStatus(
    "Ready - load model"
);