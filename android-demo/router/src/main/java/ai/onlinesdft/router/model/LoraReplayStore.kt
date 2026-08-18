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

private const val MAX_LORA_PROCESSED_FEEDBACK = 512

/** Small crash-safe companion checkpoint for replay metadata (LoRA tensors live in ORT). */
data class LoraReplayCheckpoint(
    val generation: Long,
    val modelId: String,
    val adapterUpdateIndex: Long,
    val adapterChecksum: String,
    val replay: List<LoraTrainingExample>,
    val replayRng: NumpyPcg64Snapshot,
    val processedFeedbackFingerprints: List<String>,
) {
    init {
        require(generation >= 0L)
        require(modelId.isNotBlank())
        require(adapterUpdateIndex >= 0L)
        require(adapterChecksum.isNotBlank())
        require(replay.size <= OnlineSdftLearner.REPLAY_CAPACITY)
        require(processedFeedbackFingerprints.size <= MAX_LORA_PROCESSED_FEEDBACK)
        require(processedFeedbackFingerprints.distinct().size == processedFeedbackFingerprints.size)
    }

    fun deepCopy() = copy(
        replay = replay.map(LoraTrainingExample::deepCopy),
        replayRng = replayRng.copy(),
        processedFeedbackFingerprints = processedFeedbackFingerprints.toList(),
    )
}

data class LoraReplayReceipt(val generation: Long, val checksum: String)

class LoraReplayStore(private val file: File? = null) {
    private var memory: LoraReplayCheckpoint? = null

    @Synchronized
    fun load(): LoraReplayCheckpoint? {
        if (file == null) return memory?.deepCopy()
        return slots().mapNotNull(::read).maxByOrNull { it.generation }?.deepCopy()
    }

    @Synchronized
    fun save(value: LoraReplayCheckpoint): LoraReplayReceipt {
        val committed = value.copy(
            generation = maxOf(value.generation, (load()?.generation ?: 0L) + 1L),
        ).deepCopy()
        val bytes = encode(committed)
        val checksum = sha256(bytes)
        if (file == null) {
            memory = committed
            return LoraReplayReceipt(committed.generation, checksum)
        }
        val destination = slots()[(committed.generation and 1L).toInt()]
        val directory = requireNotNull(destination.parentFile)
        require(directory.isDirectory || directory.mkdirs())
        val pending = File(directory, "${destination.name}.pending")
        FileOutputStream(pending).use { output ->
            output.write(bytes)
            output.fd.sync()
        }
        try {
            try {
                Files.move(
                    pending.toPath(),
                    destination.toPath(),
                    StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING,
                )
            } catch (_: AtomicMoveNotSupportedException) {
                Files.move(pending.toPath(), destination.toPath(), StandardCopyOption.REPLACE_EXISTING)
            }
            require(read(destination)?.generation == committed.generation)
        } finally {
            if (pending.exists()) pending.delete()
        }
        return LoraReplayReceipt(committed.generation, checksum)
    }

    @Synchronized
    fun clear() {
        memory = null
        file ?: return
        slots().forEach { slot ->
            val pending = File(requireNotNull(slot.parentFile), "${slot.name}.pending")
            require(!slot.exists() || slot.delete())
            require(!pending.exists() || pending.delete())
        }
    }

    private fun slots(): List<File> {
        val base = requireNotNull(file)
        return listOf(File("${base.path}.0"), File("${base.path}.1"))
    }

    private fun read(path: File): LoraReplayCheckpoint? = runCatching {
        if (!path.isFile || path.length() !in MIN_BYTES..MAX_BYTES) return@runCatching null
        decode(path.readBytes())
    }.getOrNull()

    private fun encode(value: LoraReplayCheckpoint): ByteArray {
        val payload = ByteArrayOutputStream().use { bytes ->
            DataOutputStream(bytes).use { output ->
                output.writeLong(value.generation)
                output.writeUTF(value.modelId)
                output.writeLong(value.adapterUpdateIndex)
                output.writeUTF(value.adapterChecksum)
                output.writeUTF(value.replayRng.state)
                output.writeUTF(value.replayRng.increment)
                output.writeInt(value.replay.size)
                value.replay.forEach { row ->
                    output.writeUTF(row.eventId)
                    output.writeUTF(row.replayLabel)
                    output.writeUTF(row.prompt)
                    output.writeInt(row.target.size)
                    row.target.forEach(output::writeDouble)
                }
                output.writeInt(value.processedFeedbackFingerprints.size)
                value.processedFeedbackFingerprints.forEach(output::writeUTF)
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

    private fun decode(encoded: ByteArray): LoraReplayCheckpoint {
        DataInputStream(ByteArrayInputStream(encoded)).use { input ->
            require(input.readInt() == MAGIC)
            require(input.readInt() == VERSION)
            val payloadSize = input.readInt()
            require(payloadSize in 1..MAX_PAYLOAD_BYTES)
            require(encoded.size == HEADER_BYTES + payloadSize + DIGEST_BYTES)
            val payload = ByteArray(payloadSize)
            input.readFully(payload)
            val expected = ByteArray(DIGEST_BYTES)
            input.readFully(expected)
            require(
                MessageDigest.isEqual(
                    expected,
                    MessageDigest.getInstance("SHA-256").digest(payload),
                ),
            )
            DataInputStream(ByteArrayInputStream(payload)).use { data ->
                val generation = data.readLong()
                val modelId = data.readUTF()
                val updateIndex = data.readLong()
                val checksum = data.readUTF()
                val rng = NumpyPcg64Snapshot(data.readUTF(), data.readUTF())
                val replaySize = data.readInt()
                require(replaySize in 0..OnlineSdftLearner.REPLAY_CAPACITY)
                val replay = List(replaySize) {
                    val eventId = data.readUTF()
                    val label = data.readUTF()
                    val prompt = data.readUTF()
                    require(data.readInt() == Route.entries.size)
                    val target = DoubleArray(Route.entries.size) { data.readDouble() }
                    LoraTrainingExample(eventId, prompt, target, label)
                }
                val processedCount = data.readInt()
                require(processedCount in 0..MAX_LORA_PROCESSED_FEEDBACK)
                val processed = List(processedCount) { data.readUTF() }
                require(data.available() == 0)
                return LoraReplayCheckpoint(
                    generation,
                    modelId,
                    updateIndex,
                    checksum,
                    replay,
                    rng,
                    processed,
                )
            }
        }
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { "%02x".format(it) }

    companion object {
        private const val MAGIC = 0x4c52504c // LRPL
        private const val VERSION = 1
        private const val HEADER_BYTES = 12
        private const val DIGEST_BYTES = 32
        private const val MAX_PAYLOAD_BYTES = 2 * 1024 * 1024
        private const val MIN_BYTES = (HEADER_BYTES + DIGEST_BYTES + 1).toLong()
        private const val MAX_BYTES = (HEADER_BYTES + DIGEST_BYTES + MAX_PAYLOAD_BYTES).toLong()
    }
}
