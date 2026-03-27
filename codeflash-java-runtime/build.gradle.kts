plugins {
    java
    id("com.gradleup.shadow") version "9.0.0-beta12"
}

group = "com.codeflash"
version = "1.0.0"

java {
    sourceCompatibility = JavaVersion.VERSION_11
    targetCompatibility = JavaVersion.VERSION_11
}

repositories {
    mavenCentral()
}

dependencies {
    implementation("com.google.code.gson:gson:2.10.1")
    implementation("com.esotericsoftware:kryo:5.6.2")
    implementation("org.objenesis:objenesis:3.4")
    implementation("org.xerial:sqlite-jdbc:3.45.0.0")
    implementation("org.ow2.asm:asm:9.7.1")
    implementation("org.ow2.asm:asm-commons:9.7.1")
    implementation("org.jacoco:org.jacoco.agent:0.8.13:runtime")
    implementation("org.jacoco:org.jacoco.cli:0.8.13:nodeps")

    testImplementation("org.junit.jupiter:junit-jupiter:5.10.1")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.test {
    useJUnitPlatform()
    jvmArgs(
        "--add-opens", "java.base/java.util=ALL-UNNAMED",
        "--add-opens", "java.base/java.lang=ALL-UNNAMED",
        "--add-opens", "java.base/java.lang.reflect=ALL-UNNAMED",
        "--add-opens", "java.base/java.math=ALL-UNNAMED",
        "--add-opens", "java.base/java.io=ALL-UNNAMED",
        "--add-opens", "java.base/java.net=ALL-UNNAMED",
        "--add-opens", "java.base/java.time=ALL-UNNAMED",
    )
}

tasks.shadowJar {
    archiveBaseName.set("codeflash-runtime")
    archiveVersion.set("1.0.0")
    archiveClassifier.set("")

    relocate("org.objectweb.asm", "com.codeflash.asm")

    manifest {
        attributes(
            "Main-Class" to "com.codeflash.Comparator",
            "Premain-Class" to "com.codeflash.profiler.ProfilerAgent",
            "Can-Retransform-Classes" to "true",
        )
    }

    exclude("META-INF/*.SF")
    exclude("META-INF/*.DSA")
    exclude("META-INF/*.RSA")
}

tasks.build {
    dependsOn(tasks.shadowJar)
}
