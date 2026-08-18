package ai.onlinesdft.router.model

import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest

/** Checksummed atomic persistence for evaluator aggregates only. */
class PrequentialMetricsStore(
    private val file: File? = null,
) {
    private var inMemory = emptyMetrics()

    @Synchronized
    fun load(): EvaluationMetrics {
        val target = file ?: return inMemory
        return runCatching { decode(target.readBytes()) }.getOrElse { emptyMetrics() }
    }

    /** Returns false on an I/O failure; scoring must remain available. */
    @Synchronized
    fun save(metrics: EvaluationMetrics): Boolean {
        validate(metrics)
        val target = file
        if (target == null) {
            inMemory = metrics
            return true
        }
        val parent = requireNotNull(target.parentFile)
        val pending = File(parent, "${target.name}.pending")
        return runCatching {
            require(parent.isDirectory || parent.mkdirs())
            val encoded = encode(metrics)
            FileOutputStream(pending).use { output ->
                output.write(encoded)
                output.fd.sync()
            }
            try {
                Files.move(
                    pending.toPath(),
                    target.toPath(),
                    StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING,
                )
            } catch (_: AtomicMoveNotSupportedException) {
                Files.move(
                    pending.toPath(),
                    target.toPath(),
                    StandardCopyOption.REPLACE_EXISTING,
                )
            }
            true
        }.getOrElse {
            pending.delete()
            false
        }
    }

    /**
     * Commits zero atomically when possible. If that write fails, removes the
     * stale target so a pre-reset aggregate cannot be resurrected on restart.
     */
    @Synchronized
    fun clear(): Boolean {
        if (save(emptyMetrics())) return true
        val target = file ?: return false
        val pending = File(target.parentFile, "${target.name}.pending")
        return runCatching {
            if (pending.isFile) pending.delete()
            !target.exists() || target.delete()
        }.getOrDefault(false)
    }

    private fun encode(metrics: EvaluationMetrics): ByteArray {
        val payload = ByteArrayOutputStream().use { bytes ->
            DataOutputStream(bytes).use { output ->
                output.writeInt(metrics.decisions)
                output.writeInt(metrics.correct)
                output.writeDouble(metrics.cumulativeRegret)
                output.writeInt(metrics.baseCorrect)
                output.writeDouble(metrics.baseCumulativeRegret)
                output.writeDouble(metrics.lastStepRegret)
            }
            bytes.toByteArray()
        }
        val digest = MessageDigest.getInstance("SHA-256").digest(payload)
        return ByteArrayOutputStream().use { bytes ->
            DataOutputStream(bytes).use { output ->
                output.writeInt(MAGIC)
                output.writeInt(VERSION)
                output.writeInt(payload.size)
                output.write(payload)
                output.write(digest)
            }
            bytes.toByteArray()
        }
    }

    private fun decode(encoded: ByteArray): EvaluationMetrics {
        DataInputStream(ByteArrayInputStream(encoded)).use { input ->
            require(input.readInt() == MAGIC)
            require(input.readInt() == VERSION)
            val size = input.readInt()
            require(size == PAYLOAD_BYTES)
            require(encoded.size == HEADER_BYTES + size + DIGEST_BYTES)
            val payload = ByteArray(size).also(input::readFully)
            val digest = ByteArray(DIGEST_BYTES).also(input::readFully)
            require(
                MessageDigest.isEqual(
                    digest,
                    MessageDigest.getInstance("SHA-256").digest(payload),
                ),
            )
            DataInputStream(ByteArrayInputStream(payload)).use { data ->
                val decisions = data.readInt()
                val correct = data.readInt()
                val cumulativeRegret = data.readDouble()
                val baseCorrect = data.readInt()
                val baseCumulativeRegret = data.readDouble()
                val lastStepRegret = data.readDouble()
                require(data.available() == 0)
                return EvaluationMetrics(
                    decisions = decisions,
                    correct = correct,
                    onlineAccuracy = accuracy(correct, decisions),
                    cumulativeRegret = cumulativeRegret,
                    baseCorrect = baseCorrect,
                    baseAccuracy = accuracy(baseCorrect, decisions),
                    baseCumulativeRegret = baseCumulativeRegret,
                    lastStepRegret = lastStepRegret,
                ).also(::validate)
            }
        }
    }

    private fun validate(metrics: EvaluationMetrics) {
        require(metrics.decisions >= 0)
        require(metrics.correct in 0..metrics.decisions)
        require(metrics.baseCorrect in 0..metrics.decisions)
        require(metrics.cumulativeRegret.isFinite() && metrics.cumulativeRegret >= 0.0)
        require(metrics.baseCumulativeRegret.isFinite() && metrics.baseCumulativeRegret >= 0.0)
        require(metrics.lastStepRegret.isFinite() && metrics.lastStepRegret >= 0.0)
    }

    companion object {
        fun emptyMetrics(): EvaluationMetrics = EvaluationMetrics(
            decisions = 0,
            correct = 0,
            onlineAccuracy = 0.0,
            cumulativeRegret = 0.0,
            baseCorrect = 0,
            baseAccuracy = 0.0,
            baseCumulativeRegret = 0.0,
            lastStepRegret = 0.0,
        )

        private fun accuracy(correct: Int, decisions: Int): Double =
            if (decisions == 0) 0.0 else correct.toDouble() / decisions

        private const val MAGIC = 0x5051534D // PQSM
        private const val VERSION = 1
        private const val PAYLOAD_BYTES = 36
        private const val HEADER_BYTES = 12
        private const val DIGEST_BYTES = 32
    }
}
