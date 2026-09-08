# Companion identity assets

Place only approved, licensed reference images here. Do not commit raw voice
recordings or provider credentials.

Paths are configured per deployment through `COMPANION_REFERENCE_IMAGE_MAP` and
are relative to this directory. Example:

```env
COMPANION_REFERENCE_IMAGE_MAP={"lina":["lina/reference-front.png","lina/reference-profile.png"]}
COMPANION_VOICE_ID_MAP={"lina":"the-elevenlabs-voice-id"}
```

The seed command merges those deployment values into each companion's database
configuration. The API exposes only `voice_available` / `image_available`; it
never returns provider voice IDs or filesystem paths.
