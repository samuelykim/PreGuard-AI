// main.cc
// W9 -- run the quantized fall-detection model directly on the ESP32-S3,
// reading real IMU sensor data instead of a CSV file.
//
// *** THIS IS A SCAFFOLD, NOT A FINISHED FILE ***
// Every section marked "TODO" needs to be filled in with your specific
// IMU sensor's actual library calls -- this can't be written correctly
// without knowing your exact sensor chip and how it's wired.
//
// IMPORTANT CAVEAT ON TFLM OPERATOR SUPPORT:
// This model uses Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense,
// and a sigmoid output. Depending on your exact esp-tflite-micro
// version, some of these ops may register under slightly different
// names (e.g. Conv1D sometimes needs to be added as a Conv2D-style op
// under the hood). Start with the "AllOpsResolver" (registers
// everything, larger firmware size but guaranteed to find your ops)
// to confirm the model loads and runs at all, THEN switch to the
// smaller MicroMutableOpResolver shown below once you know exactly
// which ops your specific model needs.

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "fall_model_data.h"           // from quantize_model.py
#include "fall_model_quant_params.h"   // from quantize_model.py
#include "normalization_params.h"      // from quantize_model.py

// TODO: replace with your actual IMU sensor's library, e.g.:
// #include "your_imu_library.h"

#define WINDOW_SIZE 50      // must match windowSize in prepare_data.py
#define NUM_FEATURES 6      // accel x,y,z + gyro x,y,z
#define STEP_SIZE 25        // 50% overlap -> half of WINDOW_SIZE
#define SAMPLE_RATE_HZ 100

// Sliding buffer holding the most recent WINDOW_SIZE readings.
float sensorBuffer[WINDOW_SIZE][NUM_FEATURES];
int samplesSinceLastInference = 0;
bool bufferFull = false;

// TFLM globals -- tensor_arena size may need tuning based on your
// actual model size (printed when quantize_model.py runs).
constexpr int kTensorArenaSize = 20 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;


// ---------------------------------------------------------
// TODO: fill this in with your specific IMU's init sequence
// ---------------------------------------------------------
void imuInit() {
    // Example shape (adjust for your actual sensor):
    // Wire.begin();
    // myImu.begin();
}


// ---------------------------------------------------------
// TODO: fill this in with your specific IMU's read call
// Must fill outValues[0..5] with: accel_x, accel_y, accel_z,
// gyro_x, gyro_y, gyro_z -- SAME ORDER as prepare_data.py used.
// ---------------------------------------------------------
void imuRead(float outValues[NUM_FEATURES]) {
    // Example shape (adjust for your actual sensor):
    // outValues[0] = myImu.getAccelX();
    // outValues[1] = myImu.getAccelY();
    // outValues[2] = myImu.getAccelZ();
    // outValues[3] = myImu.getGyroX();
    // outValues[4] = myImu.getGyroY();
    // outValues[5] = myImu.getGyroZ();
}


// ---------------------------------------------------------
// Shift the buffer left by one row and insert the new sample
// at the end -- this is what makes it a SLIDING window.
// ---------------------------------------------------------
void shiftBufferAndInsert(float newSample[NUM_FEATURES]) {
    for (int row = 0; row < WINDOW_SIZE - 1; row++) {
        for (int col = 0; col < NUM_FEATURES; col++) {
            sensorBuffer[row][col] = sensorBuffer[row + 1][col];
        }
    }
    for (int col = 0; col < NUM_FEATURES; col++) {
        sensorBuffer[WINDOW_SIZE - 1][col] = newSample[col];
    }
}


// ---------------------------------------------------------
// Normalize (same mean/std as training) then quantize to int8
// (using the scale/zero-point quantize_model.py exported).
// ---------------------------------------------------------
void normalizeAndQuantizeInto(int8_t* dest) {
    int idx = 0;
    for (int row = 0; row < WINDOW_SIZE; row++) {
        for (int col = 0; col < NUM_FEATURES; col++) {
            float normalized = (sensorBuffer[row][col] - SENSOR_MEAN[col]) / SENSOR_STD[col];
            int quantized = (int)(normalized / INPUT_SCALE) + INPUT_ZERO_POINT;

            // Clamp into valid int8 range just in case.
            if (quantized > 127) quantized = 127;
            if (quantized < -128) quantized = -128;

            dest[idx++] = (int8_t)quantized;
        }
    }
}


void setup() {
    Serial.begin(115200);
    imuInit();  // TODO

    static tflite::MicroErrorReporter micro_error_reporter;
    const tflite::Model* model = tflite::GetModel(fall_model_data);

    // Start with AllOpsResolver to confirm everything loads correctly --
    // see the caveat comment at the top of this file.
    static tflite::AllOpsResolver resolver;

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);
    interpreter = &static_interpreter;

    interpreter->AllocateTensors();

    input = interpreter->input(0);
    output = interpreter->output(0);

    Serial.println("Fall detection model loaded. Starting sensor loop...");
}


void loop() {
    float newSample[NUM_FEATURES];
    imuRead(newSample);  // TODO

    shiftBufferAndInsert(newSample);
    samplesSinceLastInference++;

    if (samplesSinceLastInference >= WINDOW_SIZE) {
        bufferFull = true;
    }

    // Run inference every STEP_SIZE new samples, once we have a full
    // window -- this is the 50% overlap sliding window in action.
    if (bufferFull && (samplesSinceLastInference % STEP_SIZE == 0)) {

        normalizeAndQuantizeInto(input->data.int8);

        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
            Serial.println("Inference failed!");
            return;
        }

        int8_t rawOutput = output->data.int8[0];
        float probability = (rawOutput - OUTPUT_ZERO_POINT) * OUTPUT_SCALE;

        Serial.print("Fall probability: ");
        Serial.print(probability * 100);
        Serial.println("%");
    }

    delay(1000 / SAMPLE_RATE_HZ);  // keep to ~100Hz sampling
}
