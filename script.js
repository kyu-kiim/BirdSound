import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const startBtn = document.getElementById("startBtn");

let faceLandmarker = null;
let animationId = null;

// 얼굴 중앙점 계산에 사용할 랜드마크 인덱스
const NOSE_TIP = 1;
const BETWEEN_EYES = 168;
const MOUTH_CENTER = 13;

function setStatus(msg) {
  statusEl.textContent = msg;
}

async function initModel() {
  setStatus("모델 준비 중...");
  try {
    const filesetResolver = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );

    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numFaces: 1,
    });

    setStatus("카메라 시작 버튼을 눌러주세요.");
    startBtn.disabled = false;
  } catch (err) {
    console.error("모델 초기화 실패:", err);
    setStatus("얼굴 인식 모델 로딩 오류. 콘솔을 확인하세요.");
  }
}

async function startCamera() {
  startBtn.disabled = true;
  setStatus("카메라 권한 요청 중...");

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
    });
    video.srcObject = stream;

    video.addEventListener("loadeddata", () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      setStatus("얼굴 중앙점 추적 중...");
      detectLoop();
    });
  } catch (err) {
    console.error("카메라 접근 실패:", err);
    setStatus("카메라 오류: 카메라 권한을 허용해주세요.");
    startBtn.disabled = false;
  }
}

function detectLoop() {
  if (!faceLandmarker) return;

  const results = faceLandmarker.detectForVideo(video, performance.now());

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (results.faceLandmarks && results.faceLandmarks.length > 0) {
    const landmarks = results.faceLandmarks[0];
    const noseTip = landmarks[NOSE_TIP];
    const betweenEyes = landmarks[BETWEEN_EYES];
    const mouthCenter = landmarks[MOUTH_CENTER];

    const centerX =
      ((noseTip.x + betweenEyes.x + mouthCenter.x) / 3) * canvas.width;
    const centerY =
      ((noseTip.y + betweenEyes.y + mouthCenter.y) / 3) * canvas.height;

    drawCenterDot(centerX, centerY);
    setStatus("얼굴 중앙점 추적 중...");
  } else {
    setStatus("얼굴을 찾는 중...");
  }

  animationId = requestAnimationFrame(detectLoop);
}

function drawCenterDot(x, y) {
  const color = "#ff4b1f";

  // 바깥 원 (반투명)
  ctx.beginPath();
  ctx.arc(x, y, 24, 0, Math.PI * 2);
  ctx.fillStyle = color + "40";
  ctx.fill();

  // 안쪽 원 (불투명)
  ctx.beginPath();
  ctx.arc(x, y, 12, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
}

startBtn.addEventListener("click", startCamera);

// 페이지 로드 시 모델 초기화
startBtn.disabled = true;
initModel();
