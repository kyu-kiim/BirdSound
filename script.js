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
let audioStarted = false;

// 시선이 특정 소리 위치에 머무는 시간을 추적 (볼륨 선형 증가용)
let dwellSoundIndex = -1;
let dwellStartTime  = 0;
const DWELL_DURATION = 4000; // ms: 70% → 90% 도달 시간

// 시계 방향 시간 위치 → sounds 폴더 파일 매핑
// angle: 12시 = 상단(0), 시계 방향으로 증가
const SOUND_DEFS = [
  { hour: 1,  file: "final_1_Chinese Blackbird.wav" },
  { hour: 3,  file: "finalize_3_Manchurian Bush Warbler .wav" },
  { hour: 5,  file: "finalize_5_Rufous-tailed Robin.wav" },
  { hour: 6,  file: "final_6_Marsh Tit.wav" },
  { hour: 7,  file: "finalize_7_Japanese bush warbler-1.wav" },
  { hour: 9,  file: "final_9_Grey-headed Woodpecker .wav" },
  { hour: 12, file: "final_12_Yellow-billed Grosbeak.wav" },
];

// 각 소리의 화면 위치와 오디오 인스턴스를 담는 배열
let sounds = [];

function computeSoundPositions() {
  const radius = Math.min(canvas.width, canvas.height) * 0.42;
  const cx = canvas.width / 2;
  const cy = canvas.height / 2;

  sounds.forEach(s => {
    const angle = (s.hour / 12) * 2 * Math.PI; // 12시 = 0, 시계방향
    s.x = cx + radius * Math.sin(angle);
    s.y = cy - radius * Math.cos(angle);
  });
}

function initAudio() {
  sounds = SOUND_DEFS.map(def => {
    const audio = new Audio(`sounds/${encodeURIComponent(def.file)}`);
    audio.loop = true;
    audio.volume = 0.5;
    return { audio, hour: def.hour, x: 0, y: 0 };
  });
  computeSoundPositions();
}

function startAudio() {
  if (audioStarted) return;
  audioStarted = true;
  permissionMsg.textContent = "";
  sounds.forEach(s => {
    s.audio.play().catch(err => console.error(`오디오 재생 실패 [${s.hour}시]:`, err));
  });
}

// 브라우저 Autoplay 정책: 오디오는 사용자 직접 인터랙션(클릭/탭) 이후에만 재생 가능.
// 첫 클릭/탭 시 오디오를 시작하고 안내 문구를 숨김.
function setupAudioUnlock() {
  const unlock = () => {
    startAudio();
    window.removeEventListener("click", unlock);
    window.removeEventListener("touchend", unlock);
  };
  window.addEventListener("click", unlock);
  window.addEventListener("touchend", unlock);
}

// 시선 위치(gazeX, gazeY)에 따라 각 소리의 볼륨을 업데이트
// - 화면 중앙(focusStrength≈0): 모든 소리 50%
// - 가장자리(focusStrength≈1): 가장 가까운 소리 70→90%, 나머지 5%
//   (해당 위치에 머물수록 70% → 90%로 선형 증가)
function updateVolumes(gazeX, gazeY) {
  const cx = canvas.width / 2;
  const cy = canvas.height / 2;

  const gazeFromCenter = Math.sqrt((gazeX - cx) ** 2 + (gazeY - cy) ** 2);
  const maxDist = Math.sqrt(cx ** 2 + cy ** 2);
  const focusStrength = Math.min(gazeFromCenter / maxDist, 1);

  // 가장 가까운 소리 인덱스
  let minDist = Infinity;
  let closestIdx = 0;
  sounds.forEach((s, i) => {
    const d = Math.sqrt((gazeX - s.x) ** 2 + (gazeY - s.y) ** 2);
    if (d < minDist) { minDist = d; closestIdx = i; }
  });

  // 중앙 근처이면 dwell 초기화
  const now = performance.now();
  if (focusStrength < 0.2) {
    dwellSoundIndex = -1;
  } else {
    if (closestIdx !== dwellSoundIndex) {
      dwellSoundIndex = closestIdx;
      dwellStartTime  = now;
    }
  }

  // 머문 시간 비율 (0 = 막 도착, 1 = DWELL_DURATION 경과)
  const dwellProgress = dwellSoundIndex >= 0
    ? Math.min((now - dwellStartTime) / DWELL_DURATION, 1)
    : 0;

  // 포커스된 소리: 70% → 90% 선형 증가 / 나머지: 5%
  const targetFocused = 0.70 + 0.20 * dwellProgress;
  const targetOther   = 0.05;
  const baseVol       = 0.50;

  sounds.forEach((s, i) => {
    const target = (i === closestIdx) ? targetFocused : targetOther;
    const vol = baseVol + (target - baseVol) * focusStrength;
    s.audio.volume = Math.max(0, Math.min(1, vol));
  });
}

function resizeCanvas() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}

window.addEventListener("resize", () => {
  resizeCanvas();
  computeSoundPositions();
});
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
  drawSoundPositions();

  if (results.faceLandmarks && results.faceLandmarks.length > 0) {
    const landmarks = results.faceLandmarks[0];

    const noseTip     = landmarks[1];
    const leftEye     = landmarks[33];
    const rightEye    = landmarks[263];
    const betweenEyes = landmarks[168];
    const mouthCenter = landmarks[13];

    // yaw: 좌우 회전
    const faceCenterX = (leftEye.x + rightEye.x) / 2;
    const eyeDistance = Math.abs(rightEye.x - leftEye.x);
    let yaw = (noseTip.x - faceCenterX) / eyeDistance;
    yaw = -yaw;

    // pitch: 상하 회전
    const faceCenterY = (betweenEyes.y + mouthCenter.y) / 2;
    const faceHeight  = Math.abs(mouthCenter.y - betweenEyes.y);
    let pitch = (noseTip.y - faceCenterY) / faceHeight;
    // pitch = -pitch; // 위아래 방향이 반대로 느껴질 경우 주석 해제

    const sensitivityX = canvas.width * 1.5;
    const sensitivityY = canvas.height * 1.5;

    let pointX = canvas.width  / 2 + yaw   * sensitivityX;
    let pointY = canvas.height / 2 + pitch * sensitivityY;

    pointX = Math.max(0, Math.min(canvas.width,  pointX));
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

    updateVolumes(smoothX, smoothY);
    drawPoint(smoothX, smoothY, 1.0);
  } else {
    console.log("얼굴을 찾는 중...");
    drawPoint(smoothX, smoothY, 0.15);
  }

  requestAnimationFrame(detectLoop);
}

// 테스트용: 각 소리의 좌표 위치에 작은 흰색 점 표시
function drawSoundPositions() {
  sounds.forEach(s => {
    ctx.beginPath();
    ctx.arc(s.x, s.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255, 255, 255, 0.45)";
    ctx.fill();
  });
}

function drawPoint(x, y, alpha) {
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

  // 중심 원
  ctx.beginPath();
  ctx.arc(x, y, 12, 0, Math.PI * 2);
  ctx.fillStyle = "#ff4b1f";
  ctx.fill();

  ctx.restore();
}

async function startApp() {
  try {
    initAudio();
    setupAudioUnlock();
    permissionMsg.textContent = "Click anywhere to start audio";
    await initModel();
    await startCamera();
  } catch (err) {
    if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
      permissionMsg.textContent = "Camera permission required";
    } else {
      console.error("앱 초기화 실패:", err);
    }
  }
}

window.addEventListener("DOMContentLoaded", startApp);
