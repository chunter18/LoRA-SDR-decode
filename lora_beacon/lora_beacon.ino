// LoRa Beacon - RFM95W (SX1276)
// Transmits "HELLO" every 3 seconds at SF12 for easy waterfall visibility
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

void setup() {
    Serial.begin(9600);
    LoRa.setPins(SS_PIN, RESET_PIN, DIO0_PIN);

    if (!LoRa.begin(915E6)) {
        Serial.println("LoRa init failed");
        while (1);
    }

    // High SF so chirps are easy to see on waterfall
    LoRa.setSpreadingFactor(12);
    LoRa.setSignalBandwidth(125E3);
    LoRa.setCodingRate4(5);
    LoRa.setTxPower(17);

    Serial.println("LoRa beacon started");
}

int packet_count = 0;

void loop() {
    LoRa.beginPacket();
    LoRa.print("HELLO");
    LoRa.endPacket();

    Serial.print("sent ");
    Serial.println(packet_count);
    delay(3000);
    packet_count++;
}
