package com.example.harmonychecker.core.harmony

import com.example.harmonychecker.TestDataLoader
import com.example.harmonychecker.TestDataLoader.matches
import com.example.harmonychecker.TestDataLoader.matchesExpected
import com.example.harmonychecker.TestDataLoader.toDomainInput
import io.kotest.matchers.collections.shouldHaveSize
import io.kotest.matchers.shouldBe
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.TestFactory
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.MethodSource
import java.io.File
import java.util.stream.Stream

/**
 * Harmony Rule Engine 參數化測試
 *
 * 使用 test-data/ 目錄中的 JSON 測試案例進行驗證。
 * 確保 Kotlin 實作與 Python 實作行為一致。
 */
@DisplayName("Harmony Rule Engine Tests")
class HarmonyRuleEngineTest {

    companion object {
        private const val TEST_DATA_ROOT = "test-data"

        /**
         * 載入指定目錄的所有測試案例
         */
        private fun loadTestCasesFromDirectory(directory: String): List<com.example.harmonychecker.TestCase> {
            val dir = File("$TEST_DATA_ROOT/$directory")
            return if (dir.exists() && dir.isDirectory) {
                TestDataLoader.loadTestDataFromDirectory(dir.absolutePath)
            } else {
                emptyList()
            }
        }

        @JvmStatic
        fun m1TestCases(): Stream<com.example.harmonychecker.TestCase> {
            return loadTestCasesFromDirectory("violations/M1_melodic_leap").stream()
        }

        @JvmStatic
        fun v1TestCases(): Stream<com.example.harmonychecker.TestCase> {
            return loadTestCasesFromDirectory("violations/V1_voice_crossing").stream()
        }

        @JvmStatic
        fun p1TestCases(): Stream<com.example.harmonychecker.TestCase> {
            return loadTestCasesFromDirectory("violations/P1_parallel_fifths").stream()
        }

        @JvmStatic
        fun p2TestCases(): Stream<com.example.harmonychecker.TestCase> {
            return loadTestCasesFromDirectory("violations/P2_hidden_intervals").stream()
        }

        @JvmStatic
        fun d1TestCases(): Stream<com.example.harmonychecker.TestCase> {
            return loadTestCasesFromDirectory("violations/D1_triad_doubling").stream()
        }

        @JvmStatic
        fun l1TestCases(): Stream<com.example.harmonychecker.TestCase> {
            return loadTestCasesFromDirectory("violations/L1_leading_tone").stream()
        }

        @JvmStatic
        fun validTestCases(): Stream<com.example.harmonychecker.TestCase> {
            return loadTestCasesFromDirectory("valid").stream()
        }

        @JvmStatic
        fun integrationTestCases(): Stream<com.example.harmonychecker.TestCase> {
            return loadTestCasesFromDirectory("integration").stream()
        }

        @JvmStatic
        fun allTestCases(): Stream<com.example.harmonychecker.TestCase> {
            return TestDataLoader.loadAllTestData(TEST_DATA_ROOT).stream()
        }
    }

    // ========== M1: Melodic Intervals ==========

    @ParameterizedTest(name = "{0}")
    @MethodSource("m1TestCases")
    @DisplayName("M1: Melodic Interval Violations")
    fun `M1 melodic intervals should match expected violations`(testCase: com.example.harmonychecker.TestCase) {
        val (chords, keySignature) = testCase.toDomainInput()
        val issues = HarmonyRuleEngine.analyze(chords, keySignature)

        val m1Issues = issues.filter { it.ruleId == "M1" }

        m1Issues.size shouldBe testCase.expectedOutput.violationCount
        if (testCase.expectedOutput.violationCount > 0) {
            m1Issues.matchesExpected(testCase.expectedOutput.violations) shouldBe true
        }
    }

    // ========== V1: Voice Crossing & Overlap ==========

    @ParameterizedTest(name = "{0}")
    @MethodSource("v1TestCases")
    @DisplayName("V1: Voice Crossing & Overlap Violations")
    fun `V1 voice crossing should match expected violations`(testCase: com.example.harmonychecker.TestCase) {
        val (chords, keySignature) = testCase.toDomainInput()
        val issues = HarmonyRuleEngine.analyze(chords, keySignature)

        val v1Issues = issues.filter { it.ruleId == "V1" }

        v1Issues.size shouldBe testCase.expectedOutput.violationCount
        if (testCase.expectedOutput.violationCount > 0) {
            v1Issues.matchesExpected(testCase.expectedOutput.violations) shouldBe true
        }
    }

    // ========== P1: Parallel Fifths & Octaves ==========

    @ParameterizedTest(name = "{0}")
    @MethodSource("p1TestCases")
    @DisplayName("P1: Parallel Fifths & Octaves Violations")
    fun `P1 parallel intervals should match expected violations`(testCase: com.example.harmonychecker.TestCase) {
        val (chords, keySignature) = testCase.toDomainInput()
        val issues = HarmonyRuleEngine.analyze(chords, keySignature)

        val p1Issues = issues.filter { it.ruleId == "P1" }

        p1Issues.size shouldBe testCase.expectedOutput.violationCount
        if (testCase.expectedOutput.violationCount > 0) {
            p1Issues.matchesExpected(testCase.expectedOutput.violations) shouldBe true
        }
    }

    // ========== P2: Hidden Fifths & Octaves ==========

    @ParameterizedTest(name = "{0}")
    @MethodSource("p2TestCases")
    @DisplayName("P2: Hidden Fifths & Octaves Violations")
    fun `P2 hidden intervals should match expected violations`(testCase: com.example.harmonychecker.TestCase) {
        val (chords, keySignature) = testCase.toDomainInput()
        val issues = HarmonyRuleEngine.analyze(chords, keySignature)

        val p2Issues = issues.filter { it.ruleId == "P2" }

        p2Issues.size shouldBe testCase.expectedOutput.violationCount
        if (testCase.expectedOutput.violationCount > 0) {
            p2Issues.matchesExpected(testCase.expectedOutput.violations) shouldBe true
        }
    }

    // ========== D1: Triad Doubling ==========

    @ParameterizedTest(name = "{0}")
    @MethodSource("d1TestCases")
    @DisplayName("D1: Triad Doubling & Omission Violations")
    fun `D1 triad doubling should match expected violations`(testCase: com.example.harmonychecker.TestCase) {
        val (chords, keySignature) = testCase.toDomainInput()
        val issues = HarmonyRuleEngine.analyze(chords, keySignature)

        val d1Issues = issues.filter { it.ruleId == "D1" }

        d1Issues.size shouldBe testCase.expectedOutput.violationCount
        if (testCase.expectedOutput.violationCount > 0) {
            d1Issues.matchesExpected(testCase.expectedOutput.violations) shouldBe true
        }
    }

    // ========== L1: Leading Tone Resolution ==========

    @ParameterizedTest(name = "{0}")
    @MethodSource("l1TestCases")
    @DisplayName("L1: Leading Tone Resolution Violations")
    fun `L1 leading tone should match expected violations`(testCase: com.example.harmonychecker.TestCase) {
        val (chords, keySignature) = testCase.toDomainInput()
        val issues = HarmonyRuleEngine.analyze(chords, keySignature)

        val l1Issues = issues.filter { it.ruleId == "L1" }

        l1Issues.size shouldBe testCase.expectedOutput.violationCount
        if (testCase.expectedOutput.violationCount > 0) {
            l1Issues.matchesExpected(testCase.expectedOutput.violations) shouldBe true
        }
    }

    // ========== Valid Cases (No Violations) ==========

    @ParameterizedTest(name = "{0}")
    @MethodSource("validTestCases")
    @DisplayName("Valid: Correct Progressions (No Violations)")
    fun `Valid progressions should have no violations`(testCase: com.example.harmonychecker.TestCase) {
        val (chords, keySignature) = testCase.toDomainInput()
        val issues = HarmonyRuleEngine.analyze(chords, keySignature)

        issues shouldHaveSize testCase.expectedOutput.violationCount
    }

    // ========== Integration Tests ==========

    @ParameterizedTest(name = "{0}")
    @MethodSource("integrationTestCases")
    @DisplayName("Integration: Mixed Violations")
    fun `Integration tests should detect all expected violations`(testCase: com.example.harmonychecker.TestCase) {
        val (chords, keySignature) = testCase.toDomainInput()
        val issues = HarmonyRuleEngine.analyze(chords, keySignature)

        issues.size shouldBe testCase.expectedOutput.violationCount

        // Verify each expected violation is found
        testCase.expectedOutput.violations.forEach { expectedViolation ->
            val matchingIssue = issues.find { it.matches(expectedViolation) }
            matchingIssue shouldBe org.junit.jupiter.api.Assertions.assertNotNull(
                matchingIssue,
                "Expected violation ${expectedViolation.ruleId} not found: ${expectedViolation.messageZhContains}"
            )
        }
    }

    // ========== Dynamic Tests for All Test Cases ==========

    @TestFactory
    @DisplayName("Dynamic Tests: All Test Cases")
    fun `all test cases should pass`(): List<DynamicTest> {
        val allCases = allTestCases().toList()

        return allCases.map { testCase ->
            DynamicTest.dynamicTest("${testCase.testId}: ${testCase.description}") {
                val (chords, keySignature) = testCase.toDomainInput()
                val issues = HarmonyRuleEngine.analyze(chords, keySignature)

                // Verify violation count
                issues.size shouldBe testCase.expectedOutput.violationCount

                // Verify each violation matches expectations
                if (testCase.expectedOutput.violationCount > 0) {
                    issues.matchesExpected(testCase.expectedOutput.violations) shouldBe true
                }
            }
        }
    }

    // ========== Specific Rule Tests ==========

    @Test
    @DisplayName("Test Case: P1_001 - S-A Parallel Fifth")
    fun `specific test - P1_001 parallel fifth`() {
        val testCase = TestDataLoader.loadTestCase("$TEST_DATA_ROOT/violations/P1_parallel_fifths/P1_001_sa_parallel_fifth.json")
        val (chords, keySignature) = testCase.toDomainInput()

        val issues = HarmonyRuleEngine.analyze(chords, keySignature)

        issues.size shouldBe 1
        issues[0].ruleId shouldBe "P1"
        issues[0].severity shouldBe Severity.ERROR
        issues[0].message.contains("平行") shouldBe true
    }

    @Test
    @DisplayName("Test Case: VALID_001 - Simple I-V-I Progression")
    fun `specific test - VALID_001 no violations`() {
        val testCase = TestDataLoader.loadTestCase("$TEST_DATA_ROOT/valid/VALID_001_simple_progression.json")
        val (chords, keySignature) = testCase.toDomainInput()

        val issues = HarmonyRuleEngine.analyze(chords, keySignature)

        issues shouldHaveSize 0
    }

    @Test
    @DisplayName("Test Case: INT_001 - Mixed Violations")
    fun `specific test - INT_001 mixed violations`() {
        val testCase = TestDataLoader.loadTestCase("$TEST_DATA_ROOT/integration/INT_001_mixed_violations.json")
        val (chords, keySignature) = testCase.toDomainInput()

        val issues = HarmonyRuleEngine.analyze(chords, keySignature)

        issues.size shouldBe 2
        issues.any { it.ruleId == "M1" } shouldBe true
        issues.any { it.ruleId == "P1" } shouldBe true
    }

    // ========== Cadence Classification Tests ==========

    @Test
    @DisplayName("Cadence: Perfect Authentic Cadence (PAC)")
    fun `should classify PAC correctly`() {
        val testCase = TestDataLoader.loadTestCase("$TEST_DATA_ROOT/valid/VALID_005_authentic_cadence.json")
        val (chords, keySignature) = testCase.toDomainInput()

        val cadence = HarmonyRuleEngine.classifyCadence(chords, keySignature)

        cadence shouldBe "PAC"
    }

    @Test
    @DisplayName("Cadence: No Cadence")
    fun `should return null for non-cadential progressions`() {
        val testCase = TestDataLoader.loadTestCase("$TEST_DATA_ROOT/violations/P1_parallel_fifths/P1_001_sa_parallel_fifth.json")
        val (chords, keySignature) = testCase.toDomainInput()

        val cadence = HarmonyRuleEngine.classifyCadence(chords, keySignature)

        cadence shouldBe null
    }

    // ========== Edge Cases ==========

    @Test
    @DisplayName("Edge Case: Empty Chord List")
    fun `should handle empty chord list gracefully`() {
        val issues = HarmonyRuleEngine.analyze(emptyList())

        issues shouldHaveSize 0
    }

    @Test
    @DisplayName("Edge Case: Single Chord")
    fun `should handle single chord`() {
        val chord = ChordSnapshot(
            index = 0,
            measure = 1,
            beat = 1.0,
            notes = mapOf(
                Voice.SOPRANO to NoteEvent(Voice.SOPRANO, 72, 1, 1.0),
                Voice.ALTO to NoteEvent(Voice.ALTO, 67, 1, 1.0),
                Voice.TENOR to NoteEvent(Voice.TENOR, 64, 1, 1.0),
                Voice.BASS to NoteEvent(Voice.BASS, 48, 1, 1.0)
            )
        )

        val issues = HarmonyRuleEngine.analyze(listOf(chord))

        // Single chord can only trigger D1 (triad doubling) rules
        issues.all { it.ruleId == "D1" || issues.isEmpty() } shouldBe true
    }
}
