// Safari / 모바일 브라우저에서는 사용자 제스처 없이 자동 카메라 요청이 차단될 수 있음.
// 해당 환경에서 작동하지 않을 경우 시작 버튼을 추가하는 방식으로 전환 필요.

import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const permissionMsg = document.getElementById("permissionMsg");

let faceLandmarker = null;
let smoothX = 0;
let smoothY = 0;
let isFirstFrame = true;
let lastDetected = false;

function resizeCanvas() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}

window.addEventListener("resize", resizeCanvas);
resizeCanvas();

async function initModel() {
  console.log("모델 준비 중...");
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

  console.log("모델 준비 완료");
}

async function startCamera() {
  console.log("카메라 권한 요청 중...");
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 1280, height: 720 },
    audio: false,
  });

  video.srcObject = stream;
  video.play();

  video.addEventListener("loadeddata", () => {
    console.log("얼굴 방향 추적 중...");
    detectLoop();
  });
}

function detectLoop() {
  if (!faceLandmarker) {
    requestAnimationFrame(detectLoop);
    return;
  }

  const results = faceLandmarker.detectForVideo(video, performance.now());

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (results.faceLandmarks && results.faceLandmarks.length > 0) {
    const landmarks = results.faceLandmarks[0];

    const noseTip = landmarks[1];
    const leftEye = landmarks[33];
    const rightEye = landmarks[263];
    const betweenEyes = landmarks[168];
    const mouthCenter = landmarks[13];

    // 좌우 회전값 yaw
    const faceCenterX = (leftEye.x + rightEye.x) / 2;
    const eyeDistance = Math.abs(rightEye.x - leftEye.x);
    let yaw = (noseTip.x - faceCenterX) / eyeDistance;
    yaw = -yaw; // 자연스러운 좌우 방향으로 반전

    // 위아래 회전값 pitch
    const faceCenterY = (betweenEyes.y + mouthCenter.y) / 2;
    const faceHeight = Math.abs(mouthCenter.y - betweenEyes.y);
    let pitch = (noseTip.y - faceCenterY) / faceHeight;
    // pitch = -pitch; // 위아래 방향이 반대로 느껴질 경우 이 줄의 주석을 해제

    const sensitivityX = canvas.width * 2.5;
    const sensitivityY = canvas.height * 2.5;

    let pointX = canvas.width / 2 + yaw * sensitivityX;
    let pointY = canvas.height / 2 + pitch * sensitivityY;

    pointX = Math.max(0, Math.min(canvas.width, pointX));
    pointY = Math.max(0, Math.min(canvas.height, pointY));

    const smoothing = 0.12;

    if (isFirstFrame) {
      smoothX = pointX;
      smoothY = pointY;
      isFirstFrame = false;
    } else {
      smoothX += (pointX - smoothX) * smoothing;
      smoothY += (pointY - smoothY) * smoothing;
    }

    lastDetected = true;
    drawPoint(smoothX, smoothY, 1.0);
  } else {
    console.log("얼굴을 찾는 중...");
    if (lastDetected) {
      // 얼굴이 사라지면 마지막 위치에 희미하게 남김
      drawPoint(smoothX, smoothY, 0.15);
    }
  }

  requestAnimationFrame(detectLoop);
}

function drawPoint(x, y, alpha) {
  const color = "#ff4b1f";

  ctx.save();
  ctx.globalAlpha = alpha;

  // 바깥 glow
  const glow = ctx.createRadialGradient(x, y, 0, x, y, 36);
  glow.addColorStop(0, "rgba(255, 75, 31, 0.25)");
  glow.addColorStop(1, "rgba(255, 75, 31, 0)");
  ctx.beginPath();
  ctx.arc(x, y, 36, 0, Math.PI * 2);
  ctx.fillStyle = glow;
  ctx.fill();

  // 바깥 링
  ctx.beginPath();
  ctx.arc(x, y, 22, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(255, 75, 31, 0.4)";
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // 중심 원
  ctx.beginPath();
  ctx.arc(x, y, 12, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();

  ctx.restore();
}

async function startApp() {
  try {
    await initModel();
    await startCamera();
  } catch (err) {
    if (
      err.name === "NotAllowedError" ||
      err.name === "PermissionDeniedError"
    ) {
      permissionMsg.textContent = "Camera permission required";
    }
    console.error("앱 초기화 실패:", err);
  }
}

window.addEventListener("DOMContentLoaded", startApp);
