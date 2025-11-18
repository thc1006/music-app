from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

VoiceName = Literal["S", "A", "T", "B"]
Severity = Literal["error", "warning"]


@dataclass
class NoteEvent:
    """
    Represent one note in one voice at a specific time.
    """
    voice: VoiceName
    midi: int
    measure: int
    beat: float


@dataclass
class ChordSnapshot:
    """
    Represent a vertical slice of four-part harmony.
    """
    index: int
    measure: int
    beat: float
    notes: Dict[VoiceName, NoteEvent]


@dataclass
class KeySignature:
    """
    Very small helper for key-dependent rules.
    """
    tonic_midi: int
    mode: Literal["major", "minor"]


@dataclass
class RuleViolation:
    rule_id: str
    message_zh: str
    detail: str
    location: Dict[str, float | int | str]
    severity: Severity


def _interval_semitones(a: int, b: int) -> int:
    """Return semitone distance b - a."""
    return b - a


def _direction(a: int, b: int) -> int:
    """Return melodic direction: 1 up, -1 down, 0 static."""
    diff = b - a
    if diff > 0:
        return 1
    if diff < 0:
        return -1
    return 0


def _is_perfect_fifth(diff: int) -> bool:
    return abs(diff) % 12 == 7


def _is_perfect_octave_or_unison(diff: int) -> bool:
    return abs(diff) % 12 == 0


def _is_step(diff: int) -> bool:
    return abs(diff) in (1, 2)


def _normalize_pc(midi: int) -> int:
    return midi % 12


@dataclass
class TriadInfo:
    root_pc: int
    quality: Literal["major", "minor", "diminished", "augmented"]
    role_by_voice: Dict[VoiceName, Literal["root", "third", "fifth"]]


def _analyze_triad(chord: ChordSnapshot) -> Optional[TriadInfo]:
    """
    Try to infer triad root and quality from four voices.
    Only handles simple 3-note triads (no sevenths / added tones).
    """
    pcs_by_voice = {v: _normalize_pc(n.midi) for v, n in chord.notes.items()}
    unique_pcs = sorted(set(pcs_by_voice.values()))
    if len(unique_pcs) < 3:
        return None

    for root_pc in unique_pcs:
        intervals = sorted((pc - root_pc) % 12 for pc in unique_pcs if pc != root_pc)
        quality: Optional[Literal["major", "minor", "diminished", "augmented"]] = None
        if intervals == [3, 7]:
            quality = "minor"
        elif intervals == [4, 7]:
            quality = "major"
        elif intervals == [3, 6]:
            quality = "diminished"
        elif intervals == [4, 8]:
            quality = "augmented"

        if quality is None:
            continue

        role_by_voice: Dict[VoiceName, Literal["root", "third", "fifth"]] = {}
        for v, pc in pcs_by_voice.items():
            d = (pc - root_pc) % 12
            if d == 0:
                role_by_voice[v] = "root"
            elif d in (3, 4):
                role_by_voice[v] = "third"
            elif d in (6, 7, 8):
                role_by_voice[v] = "fifth"
            else:
                return None

        return TriadInfo(root_pc=root_pc, quality=quality, role_by_voice=role_by_voice)

    return None


class HarmonyAnalyzer:
    """
    Main rule-checking class.

    Usage:
        analyzer = HarmonyAnalyzer(chords, key)
        violations = analyzer.analyze()
        cadence_type = analyzer.classify_cadence()
    """

    def __init__(self, chords: List[ChordSnapshot], key: Optional[KeySignature] = None):
        self.chords = chords
        self.key = key

    # ---- Public API ----

    def analyze(self) -> List[RuleViolation]:
        """
        Run all currently implemented checks.
        """
        violations: List[RuleViolation] = []
        violations.extend(self._check_melodic_intervals())
        violations.extend(self._check_voice_crossing_and_overlap())
        violations.extend(self._check_parallel_intervals())
        violations.extend(self._check_hidden_intervals())
        violations.extend(self._check_triad_doubling())
        violations.extend(self._check_leading_tone_resolution())
        return violations

    def classify_cadence(self) -> Optional[str]:
        """
        Very rough cadence classifier based on last two chords.

        Returns one of:
        - "PAC": perfect authentic cadence
        - "IAC": imperfect authentic cadence
        - "HC" : half cadence
        - "PC" : plagal cadence
        - "DC" : deceptive cadence
        - None : unknown / not enough data
        """
        if self.key is None or len(self.chords) < 2:
            return None

        last = self.chords[-1]
        prev = self.chords[-2]

        triad_last = _analyze_triad(last)
        triad_prev = _analyze_triad(prev)
        if not triad_last or not triad_prev:
            return None

        deg_last = self._scale_degree(triad_last.root_pc)
        deg_prev = self._scale_degree(triad_prev.root_pc)
        if deg_last is None or deg_prev is None:
            return None

        soprano_last = last.notes["S"].midi
        tonic_pc = _normalize_pc(self.key.tonic_midi)

        is_last_root_position = _normalize_pc(last.notes["B"].midi) == triad_last.root_pc
        is_prev_root_position = _normalize_pc(prev.notes["B"].midi) == triad_prev.root_pc
        soprano_is_tonic = _normalize_pc(soprano_last) == tonic_pc

        if deg_prev == 5 and deg_last == 1:
            if is_last_root_position and is_prev_root_position and soprano_is_tonic:
                return "PAC"
            return "IAC"

        if deg_prev == 4 and deg_last == 1:
            return "PC"

        if deg_prev == 5 and deg_last == 6:
            return "DC"

        if deg_last == 5:
            return "HC"

        return None

    # ---- Rule checks ----

    def _check_melodic_intervals(self) -> List[RuleViolation]:
        """
        A1. Melodic leap constraints.
        """
        violations: List[RuleViolation] = []
        for voice in ("S", "A", "T", "B"):
            prev_note: Optional[NoteEvent] = None
            for chord in self.chords:
                note = chord.notes[voice]
                if prev_note is not None:
                    diff = _interval_semitones(prev_note.midi, note.midi)
                    adiff = abs(diff)

                    if adiff > 12:
                        violations.append(
                            RuleViolation(
                                rule_id="M1",
                                message_zh=f"{voice} 聲部旋律跳進超過八度，違反聲部進行原則。",
                                detail=f"from {prev_note.midi} to {note.midi}, diff={adiff} semitones.",
                                location={
                                    "voice": voice,
                                    "from_measure": prev_note.measure,
                                    "from_beat": prev_note.beat,
                                    "to_measure": note.measure,
                                    "to_beat": note.beat,
                                },
                                severity="error",
                            )
                        )
                    elif adiff in (10, 11):
                        violations.append(
                            RuleViolation(
                                rule_id="M1",
                                message_zh=f"{voice} 聲部出現七度跳進，需確認後續是否有適當解決。",
                                detail=f"from {prev_note.midi} to {note.midi}, diff={adiff} semitones.",
                                location={
                                    "voice": voice,
                                    "from_measure": prev_note.measure,
                                    "from_beat": prev_note.beat,
                                    "to_measure": note.measure,
                                    "to_beat": note.beat,
                                },
                                severity="warning",
                            )
                        )
                    elif adiff == 6:
                        violations.append(
                            RuleViolation(
                                rule_id="M1",
                                message_zh=f"{voice} 聲部出現增四度／減五度的跳進，一般視為不佳旋律。",
                                detail=f"from {prev_note.midi} to {note.midi}, diff=6 semitones.",
                                location={
                                    "voice": voice,
                                    "from_measure": prev_note.measure,
                                    "from_beat": prev_note.beat,
                                    "to_measure": note.measure,
                                    "to_beat": note.beat,
                                },
                                severity="error",
                            )
                        )

                    if voice in ("A", "T") and adiff > 5:
                        violations.append(
                            RuleViolation(
                                rule_id="M1",
                                message_zh=f"{voice} 聲部跳進超過完全四度，違反內聲部平穩進行原則。",
                                detail=f"from {prev_note.midi} to {note.midi}, diff={adiff} semitones.",
                                location={
                                    "voice": voice,
                                    "from_measure": prev_note.measure,
                                    "from_beat": prev_note.beat,
                                    "to_measure": note.measure,
                                    "to_beat": note.beat,
                                },
                                severity="error",
                            )
                        )
                prev_note = note
        return violations

    def _check_voice_crossing_and_overlap(self) -> List[RuleViolation]:
        """
        A2. Voice crossing & overlap.
        """
        violations: List[RuleViolation] = []

        # Crossing in each single chord
        for chord in self.chords:
            s = chord.notes["S"].midi
            a = chord.notes["A"].midi
            t = chord.notes["T"].midi
            b = chord.notes["B"].midi

            if s < a:
                violations.append(
                    RuleViolation(
                        rule_id="V1",
                        message_zh="Soprano 低於 Alto，發生聲部交錯（Crossing）。",
                        detail=f"S={s}, A={a}",
                        location={"measure": chord.measure, "beat": chord.beat},
                        severity="error",
                    )
                )
            if a < t:
                violations.append(
                    RuleViolation(
                        rule_id="V1",
                        message_zh="Alto 低於 Tenor，發生聲部交錯（Crossing）。",
                        detail=f"A={a}, T={t}",
                        location={"measure": chord.measure, "beat": chord.beat},
                        severity="error",
                    )
                )
            if t < b:
                violations.append(
                    RuleViolation(
                        rule_id="V1",
                        message_zh="Tenor 低於 Bass，發生聲部交錯（Crossing）。",
                        detail=f"T={t}, B={b}",
                        location={"measure": chord.measure, "beat": chord.beat},
                        severity="error",
                    )
                )

        # Overlap between adjacent chords
        for i in range(len(self.chords) - 1):
            c1 = self.chords[i]
            c2 = self.chords[i + 1]

            for upper, lower in (("S", "A"), ("A", "T"), ("T", "B")):
                upper_prev = c1.notes[upper].midi
                lower_prev = c1.notes[lower].midi
                upper_next = c2.notes[upper].midi
                lower_next = c2.notes[lower].midi

                if lower_next > upper_prev:
                    violations.append(
                        RuleViolation(
                            rule_id="V1",
                            message_zh=f"{lower}/{upper} 聲部之間發生超越（Overlap）。",
                            detail=(
                                f"prev {upper}={upper_prev}, {lower}={lower_prev}; "
                                f"next {upper}={upper_next}, {lower}={lower_next}"
                            ),
                            location={"from_index": c1.index, "to_index": c2.index},
                            severity="warning",
                        )
                    )

        return violations

    def _check_parallel_intervals(self) -> List[RuleViolation]:
        """
        B1. Parallel 8ves & 5ths.
        """
        violations: List[RuleViolation] = []

        voice_pairs: List[Tuple[VoiceName, VoiceName]] = [
            ("S", "A"),
            ("S", "T"),
            ("S", "B"),
            ("A", "T"),
            ("A", "B"),
            ("T", "B"),
        ]

        for v1, v2 in voice_pairs:
            for i in range(len(self.chords) - 1):
                c1 = self.chords[i]
                c2 = self.chords[i + 1]

                n1_a = c1.notes[v1]
                n1_b = c1.notes[v2]
                n2_a = c2.notes[v1]
                n2_b = c2.notes[v2]

                interval1 = _interval_semitones(n1_b.midi, n1_a.midi)
                interval2 = _interval_semitones(n2_b.midi, n2_a.midi)

                dir1_a = _direction(n1_a.midi, n2_a.midi)
                dir1_b = _direction(n1_b.midi, n2_b.midi)

                if dir1_a == 0 or dir1_b == 0:
                    continue
                if dir1_a != dir1_b:
                    continue

                if (
                    (_is_perfect_fifth(interval1) or _is_perfect_octave_or_unison(interval1))
                    and (_is_perfect_fifth(interval2) or _is_perfect_octave_or_unison(interval2))
                ):
                    violations.append(
                        RuleViolation(
                            rule_id="P1",
                            message_zh=f"{v1}-{v2} 聲部之間出現平行八度或平行五度。",
                            detail=(
                                f"chord {c1.index}->{c2.index}, "
                                f"interval1={interval1}, interval2={interval2}"
                            ),
                            location={
                                "from_index": c1.index,
                                "to_index": c2.index,
                                "voices": f"{v1}-{v2}",
                            },
                            severity="error",
                        )
                    )

        return violations

    def _check_hidden_intervals(self) -> List[RuleViolation]:
        """
        B2. Hidden / direct 8ves & 5ths in outer voices.
        """
        violations: List[RuleViolation] = []

        for i in range(len(self.chords) - 1):
            c1 = self.chords[i]
            c2 = self.chords[i + 1]

            s1 = c1.notes["S"].midi
            b1 = c1.notes["B"].midi
            s2 = c2.notes["S"].midi
            b2 = c2.notes["B"].midi

            interval1 = _interval_semitones(b1, s1)
            interval2 = _interval_semitones(b2, s2)

            dir_s = _direction(s1, s2)
            dir_b = _direction(b1, b2)

            if dir_s == 0 or dir_b == 0 or dir_s != dir_b:
                continue

            if (not _is_perfect_fifth(interval1) and not _is_perfect_octave_or_unison(interval1)) and (
                _is_perfect_fifth(interval2) or _is_perfect_octave_or_unison(interval2)
            ):
                s_step = _is_step(_interval_semitones(s1, s2))
                b_leap = abs(_interval_semitones(b1, b2)) > 2

                if s_step and b_leap:
                    severity: Severity = "warning"
                    msg = "外聲部出現隱伏八/五度，但屬於 S 級進、B 跳進的例外情形。"
                else:
                    severity = "error"
                    msg = "外聲部出現隱伏八度或隱伏五度，需避免。"

                violations.append(
                    RuleViolation(
                        rule_id="P2",
                        message_zh=msg,
                        detail=(
                            f"chord {c1.index}->{c2.index}, "
                            f"interval1={interval1}, interval2={interval2}, "
                            f"dir_s={dir_s}, dir_b={dir_b}"
                        ),
                        location={
                            "from_index": c1.index,
                            "to_index": c2.index,
                            "voices": "S-B",
                        },
                        severity=severity,
                    )
                )

        return violations

    def _check_triad_doubling(self) -> List[RuleViolation]:
        """
        C1. Triad doubling & omission.
        """
        violations: List[RuleViolation] = []

        for chord in self.chords:
            triad = _analyze_triad(chord)
            if triad is None:
                continue

            pcs = [_normalize_pc(n.midi) for n in chord.notes.values()]
            root_count = pcs.count(triad.root_pc)

            if triad.quality in ("major", "augmented"):
                third_pc = (triad.root_pc + 4) % 12
            else:
                third_pc = (triad.root_pc + 3) % 12

            if triad.quality == "major":
                fifth_pc = (triad.root_pc + 7) % 12
            elif triad.quality == "minor":
                fifth_pc = (triad.root_pc + 7) % 12
            elif triad.quality == "diminished":
                fifth_pc = (triad.root_pc + 6) % 12
            else:
                fifth_pc = (triad.root_pc + 8) % 12

            has_root = triad.root_pc in pcs
            has_third = third_pc in pcs
            has_fifth = fifth_pc in pcs

            if not has_root or not has_third:
                violations.append(
                    RuleViolation(
                        rule_id="D1",
                        message_zh="三和弦省略了根音或三音，違反基本配置原則。",
                        detail=(
                            f"root_pc={triad.root_pc}, has_root={has_root}, "
                            f"has_third={has_third}, chord_index={chord.index}"
                        ),
                        location={"measure": chord.measure, "beat": chord.beat},
                        severity="error",
                    )
                )

            if not has_fifth and len(set(pcs)) <= 3:
                if root_count < 3:
                    violations.append(
                        RuleViolation(
                            rule_id="D1",
                            message_zh="省略五音時，四部和聲應為三根一三，根音數量不足。",
                            detail=(
                                f"root_pc={triad.root_pc}, root_count={root_count}, "
                                f"has_fifth={has_fifth}, chord_index={chord.index}"
                            ),
                            location={"measure": chord.measure, "beat": chord.beat},
                            severity="warning",
                        )
                    )

            if self.key is not None:
                tonic_pc = _normalize_pc(self.key.tonic_midi)
                leading_pc = (tonic_pc - 1) % 12
                if triad.root_pc == leading_pc and root_count > 1:
                    violations.append(
                        RuleViolation(
                            rule_id="D1",
                            message_zh="導音根音被重複，違反七級和弦常見配置原則。",
                            detail=(
                                f"leading_pc={leading_pc}, root_count={root_count}, "
                                f"chord_index={chord.index}"
                            ),
                            location={"measure": chord.measure, "beat": chord.beat},
                            severity="error",
                        )
                    )

        return violations

    def _check_leading_tone_resolution(self) -> List[RuleViolation]:
        """
        D1. Leading tone resolution.
        """
        violations: List[RuleViolation] = []
        if self.key is None:
            return violations

        tonic_pc = _normalize_pc(self.key.tonic_midi)
        leading_pc = (tonic_pc - 1) % 12

        for i in range(len(self.chords) - 1):
            c1 = self.chords[i]
            c2 = self.chords[i + 1]

            for voice in ("S", "B"):
                n1 = c1.notes[voice]
                n2 = c2.notes[voice]

                if _normalize_pc(n1.midi) == leading_pc:
                    if not (_normalize_pc(n2.midi) == tonic_pc and _interval_semitones(n1.midi, n2.midi) > 0):
                        violations.append(
                            RuleViolation(
                                rule_id="L1",
                                message_zh=f"{voice} 聲部的導音未向上解決到主音。",
                                detail=(
                                    f"leading tone MIDI={n1.midi}, next MIDI={n2.midi}, "
                                    f"expected tonic_pc={tonic_pc}"
                                ),
                                location={
                                    "voice": voice,
                                    "from_measure": n1.measure,
                                    "from_beat": n1.beat,
                                    "to_measure": n2.measure,
                                    "to_beat": n2.beat,
                                },
                                severity="error",
                            )
                        )

        return violations

    # ---- helpers ----

    def _scale_degree(self, pc: int) -> Optional[int]:
        """
        Map pitch class to scale degree 1-7 in a simple major/minor key.
        """
        if self.key is None:
            return None

        tonic_pc = _normalize_pc(self.key.tonic_midi)
        diff = (pc - tonic_pc) % 12

        mapping_major = {
            0: 1,
            2: 2,
            4: 3,
            5: 4,
            7: 5,
            9: 6,
            11: 7,
        }
        mapping_minor = {
            0: 1,
            2: 2,
            3: 3,
            5: 4,
            7: 5,
            8: 6,
            11: 7,
        }

        if self.key.mode == "major":
            return mapping_major.get(diff)
        return mapping_minor.get(diff)