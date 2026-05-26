"""
Unit tests for the semantic command parser.
"""

from __future__ import annotations

import unittest

from src.pipeline.prompt_parser import parse_command, parse_raw_background_prompt, BackgroundType


class TestPromptParser(unittest.TestCase):
    """Test suite for parsing natural language command strings into structured pipeline parameters."""

    def test_remove_only(self):
        removes, bg, solid = parse_command("remove backpack")
        self.assertEqual(removes, ["backpack"])
        self.assertIsNone(bg)
        self.assertFalse(solid)

    def test_remove_multiple(self):
        removes, bg, solid = parse_command("remove watermark, text and unwanted logo")
        self.assertEqual(removes, ["watermark", "text", "logo"])
        self.assertIsNone(bg)
        self.assertFalse(solid)

    def test_background_only(self):
        removes, bg, solid = parse_command("replace background with office")
        self.assertEqual(removes, [])
        self.assertEqual(bg, "office")
        self.assertFalse(solid)

    def test_background_alternative_pattern(self):
        removes, bg, solid = parse_command("change bg to a clean white studio room")
        self.assertEqual(removes, [])
        # Contains "white", so it should trigger solid color mode
        self.assertEqual(bg, "color:#FFFFFF")
        self.assertTrue(solid)

    def test_background_colon_pattern(self):
        removes, bg, solid = parse_command("background: cyberpunk city")
        self.assertEqual(removes, [])
        self.assertEqual(bg, "cyberpunk city")
        self.assertFalse(solid)

    def test_remove_and_make_white(self):
        removes, bg, solid = parse_command("remove id card and make background white")
        self.assertEqual(removes, ["id card"])
        self.assertEqual(bg, "color:#FFFFFF")
        self.assertTrue(solid)

    def test_change_background_to_white_and_remove_id_card(self):
        removes, bg, solid = parse_command("change background to white and remove id card")
        self.assertEqual(removes, ["id card"])
        self.assertEqual(bg, "color:#FFFFFF")
        self.assertTrue(solid)

    def test_remove_and_make_transparent(self):
        removes, bg, solid = parse_command("remove lanyard and replace background with transparent bg")
        self.assertEqual(removes, ["lanyard"])
        self.assertEqual(bg, "transparent")
        self.assertTrue(solid)

    def test_routing_failure_scenarios(self):
        # Case 1: "change background to white"
        removes, bg, solid = parse_command("change background to white")
        self.assertEqual(removes, [])
        self.assertEqual(bg, "color:#FFFFFF")
        self.assertTrue(solid)

        # Case 2: "remove background"
        removes, bg, solid = parse_command("remove background")
        self.assertEqual(removes, [])
        self.assertEqual(bg, "transparent")
        self.assertTrue(solid)

        # Case 3: "white background"
        removes, bg, solid = parse_command("white background")
        self.assertEqual(removes, [])
        self.assertEqual(bg, "color:#FFFFFF")
        self.assertTrue(solid)

        # Case 4: "studio white background"
        removes, bg, solid = parse_command("studio white background")
        self.assertEqual(removes, [])
        self.assertEqual(bg, "color:#FFFFFF")
        self.assertTrue(solid)

        # Case 5: "passport background"
        removes, bg, solid = parse_command("passport background")
        self.assertEqual(removes, [])
        self.assertEqual(bg, "color:#FFFFFF")
        self.assertTrue(solid)

        # Case 6: "studio background"
        removes, bg, solid = parse_command("studio background")
        self.assertEqual(removes, [])
        self.assertEqual(bg, "color:#FFFFFF")
        self.assertTrue(solid)

        # Case 7: "solid color background"
        removes, bg, solid = parse_command("solid color background")
        self.assertEqual(removes, [])
        self.assertEqual(bg, "color:#FFFFFF")
        self.assertTrue(solid)

    def test_custom_color_scenarios(self):
        # Blue background
        removes, bg, solid = parse_command("change background to blue")
        self.assertEqual(removes, [])
        self.assertEqual(bg, "color:#0000FF")
        self.assertTrue(solid)

        # Dark navy background
        removes, bg, solid = parse_command("dark navy background")
        self.assertEqual(removes, [])
        self.assertEqual(bg, "color:#001F3F")
        self.assertTrue(solid)

        # Light sky blue background
        removes, bg, solid = parse_command("change background to light sky blue")
        self.assertEqual(removes, [])
        self.assertEqual(bg, "color:#87CEEB")
        self.assertTrue(solid)

        # Hex code color
        removes, bg, solid = parse_command("background: #FFA500")
        self.assertEqual(removes, [])
        self.assertEqual(bg, "color:#FFA500")
        self.assertTrue(solid)

        # RGB format color
        removes, bg, solid = parse_command("set bg to rgb(128, 0, 128)")
        self.assertEqual(removes, [])
        self.assertEqual(bg, "color:rgb(128,0,128)")
        self.assertTrue(solid)

    def test_empty_or_none(self):
        self.assertEqual(parse_command(""), ([], None, False))

    def test_parse_raw_background_prompt(self):
        # 1. Solid colors
        req, col, btype = parse_raw_background_prompt("blue")
        self.assertIsNone(req)
        self.assertEqual(col, "#0000FF")
        self.assertEqual(btype, BackgroundType.SOLID_COLOR)

        req, col, btype = parse_raw_background_prompt("color:#FFA500")
        self.assertIsNone(req)
        self.assertEqual(col, "#FFA500")
        self.assertEqual(btype, BackgroundType.SOLID_COLOR)

        # 2. Studio / Portrait
        req, col, btype = parse_raw_background_prompt("clean white studio room")
        self.assertIsNone(req)
        self.assertEqual(col, "#FFFFFF")
        self.assertEqual(btype, BackgroundType.STUDIO)

        # 3. Transparent
        req, col, btype = parse_raw_background_prompt("transparent")
        self.assertIsNone(req)
        self.assertEqual(col, "transparent")
        self.assertEqual(btype, BackgroundType.TRANSPARENT)

        # 4. Generative scene
        req, col, btype = parse_raw_background_prompt("cozy coffee shop")
        self.assertEqual(req, "cozy coffee shop")
        self.assertIsNone(col)
        self.assertEqual(btype, BackgroundType.GENERATED_SCENE)

        req, col, btype = parse_raw_background_prompt("educational campus")
        self.assertEqual(req, "educational campus")
        self.assertIsNone(col)
        self.assertEqual(btype, BackgroundType.GENERATED_SCENE)


if __name__ == "__main__":
    unittest.main()
