// Cross-language conformance: read an artifact with the GENERATED TypeScript reader and
// report a checksum of every decoded section.
//
// The point is not that TypeScript can open the file. It is that both languages, given
// the same bytes, arrive at the same NUMBERS — which a CRC over the decoded arrays
// checks and a CRC over the stored bytes does not. A dtype or codec that disagrees
// across the boundary produces plausible values rather than an error, so this is the
// only place that failure can be caught.
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { crc32, decode, SCHEMA_VERSION } from "../../web/src/generated/artifact.ts";

const dir = process.argv[2];
const meta = JSON.parse(readFileSync(`${dir}/meta.json`, "utf8"));
const raw = gunzipSync(readFileSync(`${dir}/data.bin.gz`));
const buffer = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer;

const sections = decode(meta, buffer);
const out: Record<string, unknown> = { schema_version: SCHEMA_VERSION, sections: {} };
for (const entry of meta.sections) {
  const view = (sections as Record<string, ArrayBufferView>)[entry.name];
  const bytes = new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
  (out.sections as Record<string, unknown>)[entry.name] = {
    decoded_crc32: crc32(bytes),
    length: view.byteLength,
  };
}
process.stdout.write(JSON.stringify(out));
