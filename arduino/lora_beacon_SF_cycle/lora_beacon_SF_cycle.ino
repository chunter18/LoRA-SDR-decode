// LoRa Beacon - RFM95W (SX1276)
// Cycles through SF7-SF12 at fixed 125 kHz BW / 915 MHz for testing
// SF auto-detection on the SDR receiver side.
//
// Wiring (Adafruit RFM95W breakout -> Arduino UNO):
//   VIN   -> 5V (breakout has onboard regulator)
//   GND   -> GND
//   SCK   -> D13
//   MISO  -> D12
//   MOSI  -> D11
//   CS    -> D4
//   RST   -> D2
//   G0    -> D3 (IRQ/DIO0)
//
// Library: "LoRa" by Sandeep Mistry (Arduino Library Manager)

#include <SPI.h>
#include <LoRa.h>

#define SS_PIN    4
#define RESET_PIN  2
#define DIO0_PIN   3

// Each SF sends PACKETS_PER_SF packets before switching to next SF
#define PACKETS_PER_SF 5
#define PACKET_INTERVAL_MS 3000

// SF cycle: 7, 8, 9, 10, 11, 12
const int sfs[] = {7, 8, 9, 10, 11, 12};
const int N_SFS = sizeof(sfs) / sizeof(sfs[0]);

int sf_idx = 0;
int packet_count = 0;
int packets_in_sf = 0;

void applySF(int idx) {
    int sf = sfs[idx];
    LoRa.setSpreadingFactor(sf);

    Serial.print("SF");
    Serial.print(sf);
    Serial.println(" / 125kHz / 915.0 MHz");
}

void setup() {
    Serial.begin(9600);
    LoRa.setPins(SS_PIN, RESET_PIN, DIO0_PIN);

    if (!LoRa.begin(915E6)) {
        Serial.println("LoRa init failed");
        while (1);
    }

    LoRa.setSignalBandwidth(125E3);
    LoRa.setCodingRate4(5);
    LoRa.setTxPower(17);

    applySF(sf_idx);
    Serial.println("LoRa beacon started (SF cycle mode)");
}

void loop() {
    LoRa.beginPacket();
    LoRa.print("HELLO");
    LoRa.endPacket();

    Serial.print("sent #");
    Serial.print(packet_count);
    Serial.print(" (SF");
    Serial.print(sfs[sf_idx]);
    Serial.println(")");

    packet_count++;
    packets_in_sf++;

    if (packets_in_sf >= PACKETS_PER_SF) {
        packets_in_sf = 0;
        sf_idx = (sf_idx + 1) % N_SFS;
        applySF(sf_idx);
    }

    delay(PACKET_INTERVAL_MS);
}
