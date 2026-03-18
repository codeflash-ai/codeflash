plugins {
    java
    jacoco
}

group = "com.example"
version = "1.0.0"

java {
    sourceCompatibility = JavaVersion.VERSION_11
    targetCompatibility = JavaVersion.VERSION_11
}

repositories {
    mavenCentral()
    mavenLocal()
}

dependencies {
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.0")
    testImplementation("org.junit.jupiter:junit-jupiter-params:5.10.0")
    testImplementation("org.xerial:sqlite-jdbc:3.42.0.0")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
    testImplementation(files("/Users/heshammohamed/Work/codeflash/code_to_optimize/java-gradle/libs/codeflash-runtime-1.0.0.jar"))  // codeflash-runtime
}

tasks.test {
    useJUnitPlatform()
}
