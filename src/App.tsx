import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { ChangeEvent } from 'react'
import * as THREE from 'three'
import './App.css'

type WavePoint = {
  min: number
  max: number
  rms: number
}

type ChartNote = {
  id: number
  time: number
  lane: number
  intensity: number
}

type NoteResult = 'great' | 'good' | 'ok' | 'miss'
type Difficulty = 'easy' | 'normal' | 'hard' | 'expert'
type NoteSpeed = 'slow' | 'normal' | 'fast' | 'max'

type GameStats = {
  score: number
  combo: number
  maxCombo: number
  great: number
  good: number
  ok: number
  miss: number
}

type ThreeStage = {
  renderer: THREE.WebGLRenderer
  scene: THREE.Scene
  camera: THREE.PerspectiveCamera
  notesGroup: THREE.Group
  hitBar: THREE.Mesh
  pulseLight: THREE.PointLight
  animationFrame: number
  resizeObserver: ResizeObserver
}

const laneLabels = ['A', 'S', 'K', 'L']
const laneColors = ['#0d7480', '#d65141', '#f1c84b', '#6b7bdc']
const keyToLane = new Map([
  ['a', 0],
  ['s', 1],
  ['k', 2],
  ['l', 3],
])
const waveformBuckets = 1200
const difficultySettings: Record<
  Difficulty,
  { label: string; minGap: number; minIntensity: number; hitWindow: number }
> = {
  easy: { label: '簡単', minGap: 0.42, minIntensity: 0.28, hitWindow: 0.22 },
  normal: { label: '普通', minGap: 0.3, minIntensity: 0.22, hitWindow: 0.18 },
  hard: { label: '難しい', minGap: 0.22, minIntensity: 0.16, hitWindow: 0.15 },
  expert: { label: 'とても難しい', minGap: 0.16, minIntensity: 0.1, hitWindow: 0.12 },
}
const speedSettings: Record<NoteSpeed, { label: string; approachTime: number }> = {
  slow: { label: '遅い', approachTime: 2.35 },
  normal: { label: '普通', approachTime: 1.8 },
  fast: { label: '早い', approachTime: 1.25 },
  max: { label: '最大', approachTime: 0.9 },
}

const emptyStats: GameStats = {
  score: 0,
  combo: 0,
  maxCombo: 0,
  great: 0,
  good: 0,
  ok: 0,
  miss: 0,
}

function secondsLabel(value: number) {
  const minutes = Math.floor(value / 60)
  const seconds = Math.floor(value % 60)
  const fraction = Math.floor((value % 1) * 100)
  return `${minutes}:${seconds.toString().padStart(2, '0')}.${fraction
    .toString()
    .padStart(2, '0')}`
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max)
}

function mixChannels(audioBuffer: AudioBuffer) {
  const length = audioBuffer.length
  const channelCount = audioBuffer.numberOfChannels
  const samples = new Float32Array(length)

  for (let channel = 0; channel < channelCount; channel += 1) {
    const data = audioBuffer.getChannelData(channel)
    for (let index = 0; index < length; index += 1) {
      samples[index] += data[index] / channelCount
    }
  }

  return samples
}

function buildWaveform(samples: Float32Array, buckets: number) {
  const points: WavePoint[] = []
  const bucketSize = Math.max(1, Math.floor(samples.length / buckets))

  for (let start = 0; start < samples.length; start += bucketSize) {
    let min = 1
    let max = -1
    let sumSquares = 0
    const end = Math.min(start + bucketSize, samples.length)

    for (let index = start; index < end; index += 1) {
      const sample = samples[index]
      min = Math.min(min, sample)
      max = Math.max(max, sample)
      sumSquares += sample * sample
    }

    points.push({
      min,
      max,
      rms: Math.sqrt(sumSquares / Math.max(1, end - start)),
    })
  }

  return points
}

function buildChart(samples: Float32Array, sampleRate: number) {
  const frameSize = Math.max(512, Math.floor(sampleRate * 0.055))
  const energies: number[] = []

  for (let start = 0; start < samples.length; start += frameSize) {
    let sumSquares = 0
    const end = Math.min(start + frameSize, samples.length)

    for (let index = start; index < end; index += 1) {
      const sample = samples[index]
      sumSquares += sample * sample
    }

    energies.push(Math.sqrt(sumSquares / Math.max(1, end - start)))
  }

  const average =
    energies.reduce((total, energy) => total + energy, 0) /
    Math.max(1, energies.length)
  const variance =
    energies.reduce((total, energy) => total + (energy - average) ** 2, 0) /
    Math.max(1, energies.length)
  const deviation = Math.sqrt(variance)
  const threshold = average + deviation * 0.65
  const notes: ChartNote[] = []
  let lastTime = -1

  for (let index = 1; index < energies.length - 1; index += 1) {
    const energy = energies[index]
    const isPeak =
      energy > threshold &&
      energy > energies[index - 1] * 1.08 &&
      energy >= energies[index + 1]
    const time = (index * frameSize) / sampleRate

    if (!isPeak || time - lastTime < 0.16) {
      continue
    }

    const intensity = clamp(
      (energy - average) / Math.max(0.0001, deviation * 2.2),
      0.18,
      1,
    )
    const lane = Math.floor((time * 2.7 + intensity * 5) % laneLabels.length)

    notes.push({
      id: notes.length + 1,
      time,
      lane,
      intensity,
    })
    lastTime = time
  }

  return notes
}

function applyJudgment(stats: GameStats, result: NoteResult) {
  if (result === 'miss') {
    return {
      ...stats,
      combo: 0,
      miss: stats.miss + 1,
    }
  }

  const scoreByResult = {
    great: 1000,
    good: 650,
    ok: 300,
  }
  const nextCombo = stats.combo + 1

  return {
    ...stats,
    score: stats.score + scoreByResult[result] + nextCombo * 8,
    combo: nextCombo,
    maxCombo: Math.max(stats.maxCombo, nextCombo),
    [result]: stats[result] + 1,
  }
}

function applyDifficulty(notes: ChartNote[], difficulty: Difficulty) {
  const setting = difficultySettings[difficulty]
  let lastTime = -1

  return notes
    .filter((note) => {
      if (note.intensity < setting.minIntensity) {
        return false
      }

      if (note.time - lastTime < setting.minGap) {
        return false
      }

      lastTime = note.time
      return true
    })
    .map((note, index) => ({
      ...note,
      id: index + 1,
    }))
}

function resultFromOffset(
  offset: number,
  hitWindow: number,
): Exclude<NoteResult, 'miss'> {
  if (offset <= hitWindow * 0.34) {
    return 'great'
  }
  if (offset <= hitWindow * 0.67) {
    return 'good'
  }
  return 'ok'
}

function App() {
  const [fileName, setFileName] = useState('')
  const [audioUrl, setAudioUrl] = useState('')
  const [waveform, setWaveform] = useState<WavePoint[]>([])
  const [baseNotes, setBaseNotes] = useState<ChartNote[]>([])
  const [duration, setDuration] = useState(0)
  const [sampleRate, setSampleRate] = useState(0)
  const [currentTime, setCurrentTime] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [error, setError] = useState('')
  const [mode, setMode] = useState<'edit' | 'play'>('edit')
  const [gameStatus, setGameStatus] = useState<'ready' | 'playing' | 'finished'>(
    'ready',
  )
  const [gameStats, setGameStats] = useState<GameStats>(emptyStats)
  const [noteResults, setNoteResults] = useState<Record<number, NoteResult>>({})
  const [lastJudgment, setLastJudgment] = useState('Ready')
  const [difficulty, setDifficulty] = useState<Difficulty>('normal')
  const [noteSpeed, setNoteSpeed] = useState<NoteSpeed>('normal')
  const [isFullscreen, setIsFullscreen] = useState(false)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const threeStageRef = useRef<HTMLDivElement | null>(null)
  const threeRef = useRef<ThreeStage | null>(null)
  const objectUrlRef = useRef('')
  const currentTimeRef = useRef(0)
  const notesRef = useRef<ChartNote[]>([])
  const noteResultsRef = useRef<Record<number, NoteResult>>({})
  const gameStatusRef = useRef(gameStatus)
  const hitWindow = difficultySettings[difficulty].hitWindow
  const approachTime = speedSettings[noteSpeed].approachTime
  const notes = useMemo(
    () => applyDifficulty(baseNotes, difficulty),
    [baseNotes, difficulty],
  )

  const chartSummary = useMemo(() => {
    if (!notes.length || !duration) {
      return {
        density: 0,
        strongestLane: '-',
      }
    }

    const laneCounts = laneLabels.map((_, lane) =>
      notes.filter((note) => note.lane === lane).length,
    )
    const strongestLaneIndex = laneCounts.indexOf(Math.max(...laneCounts))

    return {
      density: notes.length / (duration / 60),
      strongestLane: laneLabels[strongestLaneIndex],
    }
  }, [duration, notes])

  const hitCount =
    gameStats.great + gameStats.good + gameStats.ok + gameStats.miss
  const accuracy = hitCount
    ? ((gameStats.great + gameStats.good * 0.7 + gameStats.ok * 0.35) /
        hitCount) *
      100
    : 0

  useEffect(() => {
    notesRef.current = notes
  }, [notes])

  useEffect(() => {
    noteResultsRef.current = noteResults
  }, [noteResults])

  useEffect(() => {
    gameStatusRef.current = gameStatus
  }, [gameStatus])

  useEffect(() => {
    currentTimeRef.current = currentTime
  }, [currentTime])

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(Boolean(document.fullscreenElement))
    }

    document.addEventListener('fullscreenchange', handleFullscreenChange)
    return () =>
      document.removeEventListener('fullscreenchange', handleFullscreenChange)
  }, [])

  useEffect(() => {
    return () => {
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current)
      }
    }
  }, [])

  const markNote = useCallback((noteId: number, result: NoteResult) => {
    if (noteResultsRef.current[noteId]) {
      return
    }

    noteResultsRef.current = {
      ...noteResultsRef.current,
      [noteId]: result,
    }
    setNoteResults(noteResultsRef.current)
    setGameStats((stats) => applyJudgment(stats, result))
    setLastJudgment(result.toUpperCase())
  }, [])

  const hitLane = useCallback((lane: number) => {
    if (gameStatusRef.current !== 'playing') {
      return
    }

    const time = currentTimeRef.current
    const candidate = notesRef.current
      .filter((note) => note.lane === lane && !noteResultsRef.current[note.id])
      .map((note) => ({
        note,
        offset: Math.abs(note.time - time),
      }))
      .filter(({ offset }) => offset <= hitWindow)
      .sort((a, b) => a.offset - b.offset)[0]

    if (!candidate) {
      setLastJudgment('MISS')
      setGameStats((stats) => applyJudgment(stats, 'miss'))
      return
    }

    markNote(candidate.note.id, resultFromOffset(candidate.offset, hitWindow))
  }, [hitWindow, markNote])

  useEffect(() => {
    if (!isPlaying) {
      return
    }

    let frame = 0
    const update = () => {
      const nextTime = audioRef.current?.currentTime ?? 0
      currentTimeRef.current = nextTime
      setCurrentTime(nextTime)

      if (gameStatusRef.current === 'playing') {
        notesRef.current.forEach((note) => {
          if (
            !noteResultsRef.current[note.id] &&
            note.time < nextTime - hitWindow
          ) {
            markNote(note.id, 'miss')
          }
        })
      }

      frame = requestAnimationFrame(update)
    }

    frame = requestAnimationFrame(update)
    return () => cancelAnimationFrame(frame)
  }, [hitWindow, isPlaying, markNote])

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (gameStatusRef.current !== 'playing' || event.repeat) {
        return
      }

      const lane = keyToLane.get(event.key.toLowerCase())
      if (lane === undefined) {
        return
      }

      event.preventDefault()
      hitLane(lane)
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [hitLane])

  useEffect(() => {
    const mount = threeStageRef.current
    if (!mount || mode !== 'play') {
      return
    }

    const scene = new THREE.Scene()
    scene.background = new THREE.Color('#101820')
    scene.fog = new THREE.Fog('#101820', 7, 22)

    const camera = new THREE.PerspectiveCamera(54, 1, 0.1, 100)
    camera.position.set(0, 5.4, 7.2)
    camera.lookAt(0, 0, -4.8)

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFSoftShadowMap
    mount.appendChild(renderer.domElement)

    const ambient = new THREE.AmbientLight('#dfe9f0', 0.58)
    scene.add(ambient)

    const keyLight = new THREE.DirectionalLight('#ffffff', 2.2)
    keyLight.position.set(0, 7, 4)
    keyLight.castShadow = true
    scene.add(keyLight)

    const pulseLight = new THREE.PointLight('#f1c84b', 6, 7)
    pulseLight.position.set(0, 1.2, 1.1)
    scene.add(pulseLight)

    const lanes = new THREE.Group()
    const laneWidth = 1.55
    const laneDepth = 17
    const totalWidth = laneWidth * laneLabels.length

    laneLabels.forEach((_, lane) => {
      const laneMaterial = new THREE.MeshStandardMaterial({
        color: lane % 2 === 0 ? '#172330' : '#1d2b39',
        roughness: 0.42,
        metalness: 0.08,
      })
      const laneMesh = new THREE.Mesh(
        new THREE.BoxGeometry(laneWidth * 0.94, 0.08, laneDepth),
        laneMaterial,
      )
      laneMesh.receiveShadow = true
      laneMesh.position.set(
        (lane - 1.5) * laneWidth,
        -0.04,
        -6.2,
      )
      lanes.add(laneMesh)
    })

    for (let index = 0; index <= laneLabels.length; index += 1) {
      const divider = new THREE.Mesh(
        new THREE.BoxGeometry(0.025, 0.08, laneDepth),
        new THREE.MeshStandardMaterial({
          color: '#476173',
          emissive: '#0d7480',
          emissiveIntensity: 0.18,
        }),
      )
      divider.position.set(-totalWidth / 2 + index * laneWidth, 0.01, -6.2)
      lanes.add(divider)
    }
    scene.add(lanes)

    const hitBar = new THREE.Mesh(
      new THREE.BoxGeometry(totalWidth + 0.28, 0.12, 0.18),
      new THREE.MeshStandardMaterial({
        color: '#f1c84b',
        emissive: '#f1c84b',
        emissiveIntensity: 1.4,
      }),
    )
    hitBar.position.set(0, 0.08, 1.4)
    scene.add(hitBar)

    laneLabels.forEach((_, lane) => {
      const pad = new THREE.Mesh(
        new THREE.BoxGeometry(laneWidth * 0.72, 0.18, 0.48),
        new THREE.MeshStandardMaterial({
          color: laneColors[lane],
          emissive: laneColors[lane],
          emissiveIntensity: 0.45,
          roughness: 0.28,
        }),
      )
      pad.castShadow = true
      pad.position.set((lane - 1.5) * laneWidth, 0.11, 2.08)
      scene.add(pad)
    })

    const notesGroup = new THREE.Group()
    scene.add(notesGroup)

    const resize = () => {
      const width = Math.max(320, mount.clientWidth)
      const height = Math.max(300, mount.clientHeight)
      renderer.setSize(width, height, false)
      camera.aspect = width / height
      camera.updateProjectionMatrix()
    }
    const resizeObserver = new ResizeObserver(resize)
    resizeObserver.observe(mount)
    resize()

    const render = () => {
      hitBar.rotation.z = Math.sin(performance.now() / 150) * 0.012
      pulseLight.intensity = 5 + Math.sin(performance.now() / 180) * 1.3
      renderer.render(scene, camera)
      const stage = threeRef.current
      if (stage) {
        stage.animationFrame = requestAnimationFrame(render)
      }
    }

    threeRef.current = {
      renderer,
      scene,
      camera,
      notesGroup,
      hitBar,
      pulseLight,
      animationFrame: requestAnimationFrame(render),
      resizeObserver,
    }

    return () => {
      const stage = threeRef.current
      if (stage) {
        cancelAnimationFrame(stage.animationFrame)
        stage.resizeObserver.disconnect()
        stage.scene.traverse((object) => {
          const mesh = object as THREE.Mesh
          mesh.geometry?.dispose()
          const material = mesh.material
          if (Array.isArray(material)) {
            material.forEach((item) => item.dispose())
          } else {
            material?.dispose()
          }
        })
        stage.renderer.dispose()
      }
      mount.replaceChildren()
      threeRef.current = null
    }
  }, [mode])

  useEffect(() => {
    const stage = threeRef.current
    if (!stage || mode !== 'play') {
      return
    }

    stage.notesGroup.clear()
    const laneWidth = 1.55
    const spawnZ = -13.4
    const hitZ = 1.4

    notes.forEach((note) => {
      if (noteResults[note.id]) {
        return
      }

      const untilHit = note.time - currentTime
      if (untilHit < -hitWindow || untilHit > approachTime) {
        return
      }

      const progress = 1 - untilHit / approachTime
      const mesh = new THREE.Mesh(
        new THREE.IcosahedronGeometry(0.28 + note.intensity * 0.16, 2),
        new THREE.MeshStandardMaterial({
          color: laneColors[note.lane],
          emissive: laneColors[note.lane],
          emissiveIntensity: 0.75 + note.intensity * 0.5,
          roughness: 0.22,
          metalness: 0.18,
        }),
      )
      mesh.castShadow = true
      mesh.position.set(
        (note.lane - 1.5) * laneWidth,
        0.38 + Math.sin(progress * Math.PI) * 0.3,
        spawnZ + progress * (hitZ - spawnZ),
      )
      mesh.rotation.set(progress * 4, progress * 2.8, progress * 2)
      stage.notesGroup.add(mesh)
    })
  }, [approachTime, currentTime, hitWindow, mode, noteResults, notes])

  const drawWaveform = useCallback((
    context: CanvasRenderingContext2D,
    width: number,
    height: number,
  ) => {
    const center = height * 0.48
    const gradient = context.createLinearGradient(0, 0, width, height)
    gradient.addColorStop(0, '#f8fbff')
    gradient.addColorStop(1, '#f6f4ef')
    context.fillStyle = gradient
    context.fillRect(0, 0, width, height)

    context.strokeStyle = '#d9dee7'
    context.lineWidth = 1
    for (let line = 0; line <= 4; line += 1) {
      const y = (height / 4) * line
      context.beginPath()
      context.moveTo(0, y)
      context.lineTo(width, y)
      context.stroke()
    }

    if (waveform.length) {
      const step = width / waveform.length
      context.strokeStyle = '#16202d'
      context.lineWidth = 1.2
      context.beginPath()
      waveform.forEach((point, index) => {
        const x = index * step
        const top = center + point.max * -center * 0.82
        const bottom = center + point.min * -center * 0.82
        context.moveTo(x, top)
        context.lineTo(x, bottom)
      })
      context.stroke()

      context.fillStyle = 'rgba(13, 116, 128, 0.18)'
      waveform.forEach((point, index) => {
        const x = index * step
        context.fillRect(x, center - point.rms * center, step, point.rms * center)
      })
    }

    notes.forEach((note) => {
      const x = duration ? (note.time / duration) * width : 0
      const laneHeight = height / laneLabels.length
      const y = note.lane * laneHeight + laneHeight / 2
      context.fillStyle = `rgba(214, 81, 65, ${0.35 + note.intensity * 0.55})`
      context.beginPath()
      context.arc(x, y, 3 + note.intensity * 5, 0, Math.PI * 2)
      context.fill()
    })

    const progress = duration ? currentTime / duration : 0
    const playheadX = clamp(progress, 0, 1) * width
    context.strokeStyle = '#0d7480'
    context.lineWidth = 2
    context.beginPath()
    context.moveTo(playheadX, 0)
    context.lineTo(playheadX, height)
    context.stroke()
  }, [currentTime, duration, notes, waveform])

  useEffect(() => {
    if (mode === 'play') {
      return
    }

    const canvas = canvasRef.current
    if (!canvas) {
      return
    }

    const context = canvas.getContext('2d')
    if (!context) {
      return
    }

    const rect = canvas.getBoundingClientRect()
    const ratio = window.devicePixelRatio || 1
    canvas.width = Math.floor(rect.width * ratio)
    canvas.height = Math.floor(rect.height * ratio)
    context.scale(ratio, ratio)

    const width = rect.width
    const height = rect.height
    context.clearRect(0, 0, width, height)

    drawWaveform(context, width, height)
  }, [drawWaveform, mode])

  async function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    if (!file) {
      return
    }

    setIsAnalyzing(true)
    setError('')
    setFileName(file.name)
    resetGame()
    setCurrentTime(0)
    setIsPlaying(false)

    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current)
    }

    const nextUrl = URL.createObjectURL(file)
    objectUrlRef.current = nextUrl
    setAudioUrl(nextUrl)

    try {
      const arrayBuffer = await file.arrayBuffer()
      const audioContext = new AudioContext()
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0))
      await audioContext.close()
      const samples = mixChannels(audioBuffer)

      setDuration(audioBuffer.duration)
      setSampleRate(audioBuffer.sampleRate)
      setWaveform(buildWaveform(samples, waveformBuckets))
      setBaseNotes(buildChart(samples, audioBuffer.sampleRate))
    } catch {
      setError('音声を読み込めませんでした。mp3、wav、ogg など別の形式で試してください。')
      setWaveform([])
      setBaseNotes([])
      setDuration(0)
      setSampleRate(0)
    } finally {
      setIsAnalyzing(false)
    }
  }

  function resetGame() {
    noteResultsRef.current = {}
    setNoteResults({})
    setGameStats(emptyStats)
    setGameStatus('ready')
    setLastJudgment('Ready')
  }

  function togglePlayback() {
    const audio = audioRef.current
    if (!audio || !audioUrl) {
      return
    }

    if (audio.paused) {
      void audio.play()
      setIsPlaying(true)
    } else {
      audio.pause()
      setIsPlaying(false)
    }
  }

  function startGame() {
    const audio = audioRef.current
    if (!audio || !notes.length) {
      return
    }

    resetGame()
    setMode('play')
    setGameStatus('playing')
    audio.currentTime = 0
    currentTimeRef.current = 0
    setCurrentTime(0)
    void audio.play()
    setIsPlaying(true)
  }

  function stopGame() {
    audioRef.current?.pause()
    setIsPlaying(false)
    setGameStatus('ready')
  }

  async function fullScreenGame() {
    const targetElement = document.getElementById('waveform-panel')
    if (!targetElement) {
      return
    }

    if (!document.fullscreenElement) {
      await targetElement.requestFullscreen()
    } else {
      await document.exitFullscreen()
    }
  }

  function changeDifficulty(nextDifficulty: Difficulty) {
    audioRef.current?.pause()
    if (audioRef.current) {
      audioRef.current.currentTime = 0
    }
    setIsPlaying(false)
    setCurrentTime(0)
    setDifficulty(nextDifficulty)
    resetGame()
  }

  function changeNoteSpeed(nextSpeed: NoteSpeed) {
    setNoteSpeed(nextSpeed)
  }

  function seekFromCanvas(event: React.MouseEvent<HTMLCanvasElement>) {
    if (!duration || !audioRef.current || mode === 'play') {
      return
    }

    const rect = event.currentTarget.getBoundingClientRect()
    const nextTime = clamp((event.clientX - rect.left) / rect.width, 0, 1) * duration
    audioRef.current.currentTime = nextTime
    setCurrentTime(nextTime)
  }

  function hitLaneFromPointer(event: React.PointerEvent<HTMLDivElement>) {
    if (mode !== 'play' || gameStatus !== 'playing') {
      return
    }

    event.preventDefault()
    event.currentTarget.setPointerCapture(event.pointerId)
    const rect = event.currentTarget.getBoundingClientRect()
    const lane = clamp(
      Math.floor(((event.clientX - rect.left) / rect.width) * laneLabels.length),
      0,
      laneLabels.length - 1,
    )
    hitLane(lane)
  }

  function exportChart() {
    const chart = {
      title: fileName || 'untitled',
      duration,
      sampleRate,
      difficulty,
      speed: noteSpeed,
      lanes: laneLabels,
      notes: notes.map((note) => ({
        time: Number(note.time.toFixed(3)),
        lane: note.lane,
        key: laneLabels[note.lane],
        intensity: Number(note.intensity.toFixed(3)),
      })),
    }
    const blob = new Blob([JSON.stringify(chart, null, 2)], {
      type: 'application/json',
    })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `${fileName.replace(/\.[^.]+$/, '') || 'rhythm-chart'}.json`
    anchor.click()
    URL.revokeObjectURL(url)
  }

  return (
    <main className="app-shell">
      <section className="top-bar">
        <div>
          <p className="eyebrow">rhythm.sharkbot.me</p>
          <h1>音楽から譜面を作って遊ぶ</h1>
        </div>
        <label className="upload-button">
          <input accept="audio/*" type="file" onChange={handleFileChange} />
          音楽をアップロード
        </label>
      </section>

      <section className="workspace">
        <div className={`waveform-panel ${mode === 'play' ? 'play-panel' : ''}`} id='waveform-panel'>
          <div className="panel-header">
            <div>
              <h2>{fileName || '音声ファイルを選択してください'}</h2>
              <p>
                {duration
                  ? `${secondsLabel(duration)} / ${sampleRate.toLocaleString()} Hz`
                  : '波形、ノーツ、プレイ画面は読み込み後に表示されます。'}
              </p>
            </div>
            <div className="transport">
              <div className="mode-switch" aria-label="表示モード">
                <button
                  type="button"
                  className={mode === 'edit' ? 'active' : ''}
                  onClick={() => setMode('edit')}
                >
                  編集
                </button>
                <button
                  type="button"
                  className={mode === 'play' ? 'active' : ''}
                  onClick={() => setMode('play')}
                >
                  遊ぶ
                </button>
              </div>
              <button type="button" onClick={togglePlayback} disabled={!audioUrl}>
                {isPlaying ? '||' : '>'}
              </button>
              <span>{secondsLabel(currentTime)}</span>
            </div>
          </div>

          {mode === 'play' ? (
            <div
              className="game-stage-wrap"
              onPointerDown={hitLaneFromPointer}
              role="application"
              tabIndex={0}
            >
              <div
                ref={threeStageRef}
                className="game-stage"
                aria-label="3Dゲームレーン"
              />
              <div className="score-overlay" aria-live="polite">
                <span>{gameStatus === 'finished' ? 'Result' : lastJudgment}</span>
                <strong>{gameStats.score.toLocaleString()}</strong>
                <p>{gameStats.combo} combo</p>
              </div>
              <div className="lane-label-overlay" aria-hidden="true">
                {laneLabels.map((label) => (
                  <span key={label}>{label}</span>
                ))}
              </div>
            </div>
          ) : (
            <canvas
              ref={canvasRef}
              className="waveform-canvas"
              onClick={seekFromCanvas}
              aria-label="音声波形"
            />
          )}

          {isAnalyzing && <p className="status">解析中です...</p>}
          {error && <p className="error">{error}</p>}

          <div className="play-settings">
            <label>
              <span>Difficulty</span>
              <select
                value={difficulty}
                onChange={(event) =>
                  changeDifficulty(event.currentTarget.value as Difficulty)
                }
              >
                {Object.entries(difficultySettings).map(([value, setting]) => (
                  <option key={value} value={value}>
                    {setting.label}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span>スピード</span>
              <select
                value={noteSpeed}
                onChange={(event) =>
                  changeNoteSpeed(event.currentTarget.value as NoteSpeed)
                }
              >
                {Object.entries(speedSettings).map(([value, setting]) => (
                  <option key={value} value={value}>
                    {setting.label}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className="game-actions">
            <button type="button" onClick={startGame} disabled={!notes.length}>
              ゲーム開始
            </button>
            <button type="button" onClick={stopGame} disabled={gameStatus !== 'playing'}>
              停止
            </button>
            <button type="button" onClick={fullScreenGame}>
              {isFullscreen ? "フルスクリーンの解除" : "フルスクリーン化"}
            </button>
            <span>A / S / K / L でノーツを叩く</span>
          </div>

          <audio
            ref={audioRef}
            src={audioUrl}
            onPause={() => setIsPlaying(false)}
            onEnded={() => {
              setIsPlaying(false)
              setGameStatus(gameStatusRef.current === 'playing' ? 'finished' : 'ready')
            }}
            onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
          />
        </div>

        <aside className="chart-panel">
          <div className="score-board">
            <span>{gameStatus === 'finished' ? 'Result' : lastJudgment}</span>
            <strong>{gameStats.score.toLocaleString()}</strong>
            <p>
              {gameStats.combo} combo / {accuracy.toFixed(1)}%
            </p>
          </div>

          <div className="metric-grid">
            <div>
              <span>Notes</span>
              <strong>{notes.length}</strong>
            </div>
            <div>
              <span>Density</span>
              <strong>{chartSummary.density.toFixed(1)}/min</strong>
            </div>
            <div>
              <span>Main lane</span>
              <strong>{chartSummary.strongestLane}</strong>
            </div>
          </div>

          <div className="judgment-grid">
            <div>
              <span>Great</span>
              <strong>{gameStats.great}</strong>
            </div>
            <div>
              <span>Good</span>
              <strong>{gameStats.good}</strong>
            </div>
            <div>
              <span>OK</span>
              <strong>{gameStats.ok}</strong>
            </div>
            <div>
              <span>Miss</span>
              <strong>{gameStats.miss}</strong>
            </div>
          </div>

          <div className="lanes">
            {laneLabels.map((label) => (
              <div key={label} className="lane">
                <span>{label}</span>
              </div>
            ))}
          </div>

          <div className="note-list">
            <div className="note-list-header">
              <h2>生成ノーツ</h2>
              <button type="button" onClick={exportChart} disabled={!notes.length}>
                JSON
              </button>
            </div>
            {notes.slice(0, 10).map((note) => (
              <div key={note.id} className="note-row">
                <span>{secondsLabel(note.time)}</span>
                <strong>{laneLabels[note.lane]}</strong>
                <meter min="0" max="1" value={note.intensity} />
              </div>
            ))}
            {!notes.length && (
              <p className="empty">
                アップロード後、自動生成された譜面がここに並びます。
              </p>
            )}
          </div>
        </aside>
      </section>
    </main>
  )
}

export default App
