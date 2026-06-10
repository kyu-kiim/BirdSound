// Safari / 모바일 브라우저에서는 사용자 제스처 없이 자동 카메라 요청이 차단될 수 있음.
// 해당 환경에서 작동하지 않을 경우 시작 버튼을 추가하는 방식으로 전환 필요.

import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

const video       = document.getElementById("webcam");
const canvas      = document.getElementById("overlay");
const ctx         = canvas.getContext("2d");
const permissionMsg = document.getElementById("permissionMsg");

let faceLandmarker = null;
let smoothX = 0;
let smoothY = 0;
let isFirstFrame = true;
let audioStarted = false;

let dwellSoundIndex = -1;
let dwellStartTime  = 0;
const DWELL_DURATION = 4000; // ms: 70% → 90% 도달 시간

const SOUND_DEFS = [
  { hour: 1,  name: "Chinese Blackbird",        file: "final_1_Chinese Blackbird.wav" },
  { hour: 3,  name: "Manchurian Bush Warbler",  file: "finalize_3_Manchurian Bush Warbler .wav" },
  { hour: 5,  name: "Rufous-tailed Robin",      file: "finalize_5_Rufous-tailed Robin.wav" },
  { hour: 6,  name: "Marsh Tit",                file: "final_6_Marsh Tit.wav" },
  { hour: 7,  name: "Japanese Bush Warbler",    file: "finalize_7_Japanese bush warbler-1.wav" },
  { hour: 9,  name: "Grey-headed Woodpecker",   file: "final_9_Grey-headed Woodpecker .wav" },
  { hour: 12, name: "Yellow-billed Grosbeak",   file: "final_12_Yellow-billed Grosbeak.wav" },
];

let sounds = [];

// 시계 방향(hour) → 화면 가장자리 좌표 투영
// padding: 1.0이면 정확한 엣지, 0.88이면 약간 안쪽
function edgePosition(hour, cx, cy, w, h, padding = 0.90) {
  const angle = (hour / 12) * 2 * Math.PI;
  const dx = Math.sin(angle);
  const dy = -Math.cos(angle);

  let t = Infinity;
  if (dx > 0) t = Math.min(t, (w - cx) / dx);
  if (dx < 0) t = Math.min(t, -cx / dx);
  if (dy > 0) t = Math.min(t, (h - cy) / dy);
  if (dy < 0) t = Math.min(t, -cy / dy);

  return {
    x: cx + t * dx * padding,
    y: cy + t * dy * padding,
  };
}

function computeSoundPositions() {
  const cx = canvas.width  / 2;
  const cy = canvas.height / 2;

  sounds.forEach(s => {
    const pos = edgePosition(s.hour, cx, cy, canvas.width, canvas.height);
    s.x = pos.x;
    s.y = pos.y;
  });
}

function initAudio() {
  sounds = SOUND_DEFS.map(def => {
    const audio = new Audio(`sounds/${encodeURIComponent(def.file)}`);
    audio.loop = true;
    audio.volume = 0.5;
    return { audio, hour: def.hour, name: def.name, x: 0, y: 0 };
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

// 브라우저 Autoplay 정책: 첫 클릭/탭 시 오디오 시작
function setupAudioUnlock() {
  const unlock = () => {
    startAudio();
    window.removeEventListener("click",    unlock);
    window.removeEventListener("touchend", unlock);
  };
  window.addEventListener("click",    unlock);
  window.addEventListener("touchend", unlock);
}

// 볼륨 업데이트
// - 중앙 영역(focusStrength < 0.25): 모든 소리 50%로 복귀, dwell 초기화
// - 소리 구역 진입: 해당 소리 70% → 90% (dwell 선형 증가), 나머지 5%
// - 소리 구역 내에 머무는 한 볼륨 유지 (focusStrength 크기와 무관)
function updateVolumes(gazeX, gazeY) {
  const now = performance.now();
  const cx  = canvas.width  / 2;
  const cy  = canvas.height / 2;

  const gazeFromCenter = Math.sqrt((gazeX - cx) ** 2 + (gazeY - cy) ** 2);
  const maxDist        = Math.sqrt(cx ** 2 + cy ** 2);
  const focusStrength  = Math.min(gazeFromCenter / maxDist, 1);

  // 가장 가까운 소리
  let minDist = Infinity, closestIdx = 0;
  sounds.forEach((s, i) => {
    const d = Math.sqrt((gazeX - s.x) ** 2 + (gazeY - s.y) ** 2);
    if (d < minDist) { minDist = d; closestIdx = i; }
  });

  if (focusStrength < 0.25) {
    // 중앙 영역: dwell 초기화 + 50%로 부드럽게 복귀
    dwellSoundIndex = -1;
    sounds.forEach(s => {
      s.audio.volume += (0.5 - s.audio.volume) * 0.06;
    });
    return;
  }

  // 소리 구역 진입/이동: dwell 타이머 갱신
  if (closestIdx !== dwellSoundIndex) {
    dwellSoundIndex = closestIdx;
    dwellStartTime  = now;
  }

  const dwellProgress  = Math.min((now - dwellStartTime) / DWELL_DURATION, 1);
  const targetFocused  = 0.70 + 0.20 * dwellProgress; // 70% → 90%

  sounds.forEach((s, i) => {
    const target = (i === closestIdx) ? targetFocused : 0.05;
    // 부드러운 전환 (약 0.06 factor ≒ 1초 내 수렴)
    s.audio.volume += (target - s.audio.volume) * 0.06;
    s.audio.volume  = Math.max(0, Math.min(1, s.audio.volume));
  });
}

function resizeCanvas() {
  canvas.width  = window.innerWidth;
  canvas.height = window.innerHeight;
}

window.addEventListener("resize", () => {
  resizeCanvas();
  computeSoundPositions();
});
resizeCanvas();

// 테스트용: 소리 위치에 점 + 이름 표시
function drawSoundPositions() {
  const cx = canvas.width  / 2;
  const cy = canvas.height / 2;
  const LABEL_OFFSET = 18;

  sounds.forEach(s => {
    // 중심에서 소리 방향 단위벡터
    const dx  = s.x - cx;
    const dy  = s.y - cy;
    const len = Math.sqrt(dx * dx + dy * dy);
    const nx  = dx / len;
    const ny  = dy / len;

    // 위치 점
    ctx.beginPath();
    ctx.arc(s.x, s.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255, 255, 255, 0.5)";
    ctx.fill();

    // 레이블 (점에서 중앙 반대 방향으로 오프셋)
    const lx = s.x + nx * LABEL_OFFSET;
    const ly = s.y + ny * LABEL_OFFSET;

    ctx.font         = "11px 'Segoe UI', sans-serif";
    ctx.fillStyle    = "rgba(255, 255, 255, 0.55)";
    ctx.textBaseline = Math.abs(nx) >= Math.abs(ny)
      ? "middle"
      : (ny > 0 ? "top" : "bottom");
    ctx.textAlign    = Math.abs(nx) >= Math.abs(ny)
      ? (nx > 0 ? "left" : "right")
      : "center";

    ctx.fillText(s.name, lx, ly);
  });
}

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

    const faceCenterX = (leftEye.x + rightEye.x) / 2;
    const eyeDistance = Math.abs(rightEye.x - leftEye.x);
    let yaw = (noseTip.x - faceCenterX) / eyeDistance;
    yaw = -yaw;

    const faceCenterY = (betweenEyes.y + mouthCenter.y) / 2;
    const faceHeight  = Math.abs(mouthCenter.y - betweenEyes.y);
    let pitch = (noseTip.y - faceCenterY) / faceHeight;
    // pitch = -pitch; // 위아래 반전 필요 시 주석 해제

    const sensitivityX = canvas.width  * 1.5;
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

function drawPoint(x, y, alpha) {
  ctx.save();
  ctx.globalAlpha = alpha;

  const glow = ctx.createRadialGradient(x, y, 0, x, y, 36);
  glow.addColorStop(0, "rgba(255, 75, 31, 0.25)");
  glow.addColorStop(1, "rgba(255, 75, 31, 0)");
  ctx.beginPath();
  ctx.arc(x, y, 36, 0, Math.PI * 2);
  ctx.fillStyle = glow;
  ctx.fill();

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
