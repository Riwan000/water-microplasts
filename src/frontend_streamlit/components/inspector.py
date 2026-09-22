"""Clickable particle inspector gallery + 'Unclassified Debris' review tab."""

from PIL import Image
import streamlit as st

_GALLERY_COLUMNS = 4


def render_placeholder() -> None:
    st.caption("Populated once Step 4 inference results are wired in.")


def _crop(image: Image.Image, bbox: list[float]) -> Image.Image:
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(image.width, int(x2)), min(image.height, int(y2))
    if x2 <= x1 or y2 <= y1:
        return image.crop((0, 0, min(1, image.width), min(1, image.height)))
    return image.crop((x1, y1, x2, y2))


def _render_gallery(image: Image.Image, particles: list[dict], empty_message: str) -> None:
    if not particles:
        st.info(empty_message)
        return

    columns = st.columns(_GALLERY_COLUMNS)
    for idx, particle in enumerate(particles):
        col = columns[idx % _GALLERY_COLUMNS]
        with col:
            crop = _crop(image, particle.get("bbox", [0, 0, 1, 1]))
            st.image(crop, use_container_width=True)
            st.caption(f"**{particle.get('class_name', 'unknown')}** · {particle.get('confidence', 0):.2f}")
            with st.expander(f"Particle {particle.get('particle_id', idx)}"):
                st.write(f"Length: {particle.get('length_um', 0):.1f} µm")
                st.write(f"Width: {particle.get('width_um', 0):.1f} µm")
                st.write(f"Aspect ratio: {particle.get('aspect_ratio', 0):.2f}")
                st.write(f"Surface area: {particle.get('surface_area_um2', 0):.1f} µm²")
                if "in_focus" in particle:
                    st.write("In focus: " + ("yes" if particle["in_focus"] else "no"))


def render(image: Image.Image, confirmed: list[dict], quarantine: list[dict], height: int = 420) -> None:
    """Render the particle inspector gallery with a quarantine review tab.

    Args:
        image: The original uploaded PIL image (bboxes are in its pixel space).
        confirmed: Detections that passed the confidence quarantine gate.
        quarantine: Low-confidence detections routed to "Unclassified Debris".
        height: Fixed pixel height of the scrollable gallery area — keeps a
            large particle count from growing the overall page height.
    """
    confirmed_tab, quarantine_tab = st.tabs([
        f"Confirmed Particles ({len(confirmed)})",
        f"Unclassified Debris ({len(quarantine)})",
    ])
    with confirmed_tab:
        with st.container(height=height):
            _render_gallery(image, confirmed, "No confirmed particles in this sample.")
    with quarantine_tab:
        st.caption("Below the confidence quarantine gate — review before trusting the classification.")
        with st.container(height=height):
            _render_gallery(image, quarantine, "Nothing was quarantined in this sample.")
