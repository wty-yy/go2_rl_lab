"""Encode Isaac Lab's configuration-owned video streams with imageio-ffmpeg."""


class FFmpegImageSequenceClip:
    """Provide the frame encoder interface used by Isaac Lab's VideoRecorder."""

    def __init__(self, frames, fps):
        """Retain the captured RGB frames and their configured playback rate."""
        self.frames = frames
        self.fps = fps

    def write_videofile(self, path, codec="libx264", audio=False, logger=None):
        """Write the captured frames and close FFmpeg even when encoding fails."""
        import imageio_ffmpeg
        import numpy as np

        height, width = self.frames[0].shape[:2]
        writer = imageio_ffmpeg.write_frames(
            path,
            (width, height),
            fps=self.fps,
            codec=codec,
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            macro_block_size=1,
            ffmpeg_log_level="error",
            output_params=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"],
        )
        try:
            next(writer)
            for frame in self.frames:
                writer.send(np.ascontiguousarray(frame[..., :3], dtype=np.uint8))
        finally:
            writer.close()


def configure_video_writer() -> None:
    """Attach the FFmpeg encoder after the runtime is initialized.

    Isaac Lab 3.0's recorder imports MoviePy 1.x, whose dependency constraints conflict with
    the image's IPython stack. Only the encoding adapter changes; Isaac Lab still
    owns capture sources, clip scheduling, filenames, and recorder cleanup.
    """
    try:
        import imageio_ffmpeg  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Video recording requires the video extra: python -m pip install -e 'source/go2_rl_lab[video]'"
        ) from exc
    from isaaclab.envs.utils import video_recorder

    video_recorder.ImageSequenceClip = FFmpegImageSequenceClip


def configure_video_capture(env_cfg, args_cli) -> None:
    """Use headless Newton GL for automatic recording on either physics backend."""

    if (
        args_cli.video
        and not getattr(args_cli, "visualizer_explicit", False)
        and not env_cfg.sim.visualizer_cfgs
    ):
        from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

        env_cfg.sim.visualizer_cfgs = [NewtonGLVisualizerCfg(headless=True)]
