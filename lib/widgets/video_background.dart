import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

class VideoBackground extends StatefulWidget {
  final String assetPath;
  final Widget child;
  /// Dark overlay on top of the video. Default is 0 (no shade) — the
  /// video shows at full natural brightness/color. Only raise this on
  /// a specific screen if text directly on the video (no glass card
  /// behind it) becomes hard to read.
  final double overlayOpacity;

  /// If set, wraps [child] in a rounded rectangle inset by this margin,
  /// so the full-bleed video peeks out around the edges instead of the
  /// content covering the whole screen. This is what gives the
  /// "floating rounded card over the video" look.
  final EdgeInsetsGeometry? cardMargin;

  /// Corner radius for the card described above. Only used if [cardMargin]
  /// is set.
  final double cardRadius;

  /// Background color of the card described above. Defaults to
  /// transparent so the video still shows through unless the screen's
  /// own content (e.g. a GlassCard) provides its own backing.
  final Color cardColor;

  const VideoBackground({
    super.key,
    required this.assetPath,
    required this.child,
    this.overlayOpacity = 0.0,
    this.cardMargin,
    this.cardRadius = 28,
    this.cardColor = Colors.transparent,
  });

  @override
  State<VideoBackground> createState() => _VideoBackgroundState();
}

class _VideoBackgroundState extends State<VideoBackground> {
  late final VideoPlayerController _controller;

  @override
  void initState() {
    super.initState();
    _controller = VideoPlayerController.asset(widget.assetPath)
      ..setLooping(true)
      ..setVolume(0)
      ..initialize().then((_) {
        if (!mounted) return;
        setState(() {});
        _controller.play();
      });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    Widget content = widget.child;

    // If a card margin is given, inset the content and clip it into a
    // rounded rectangle sitting on top of the full-bleed video below.
    if (widget.cardMargin != null) {
      content = Padding(
        padding: widget.cardMargin!,
        child: ClipRRect(
          borderRadius: BorderRadius.circular(widget.cardRadius),
          child: Container(
            color: widget.cardColor,
            child: widget.child,
          ),
        ),
      );
    }

    return Stack(
      fit: StackFit.expand,
      children: [
        // Full-bleed video. Scale is computed explicitly so the video
        // always covers 100% of the screen edge-to-edge — no black
        // strip at the top or bottom, regardless of the video's own
        // aspect ratio vs the phone's aspect ratio.
        Positioned.fill(
          child: _controller.value.isInitialized
              ? LayoutBuilder(
                  builder: (context, constraints) {
                    final videoW = _controller.value.size.width;
                    final videoH = _controller.value.size.height;
                    final boxW = constraints.maxWidth;
                    final boxH = constraints.maxHeight;

                    // Cover = scale so BOTH dimensions meet or exceed
                    // the container — this is what guarantees zero gap.
                    final scale = (boxW / videoW) > (boxH / videoH)
                        ? boxW / videoW
                        : boxH / videoH;

                    return ClipRect(
                      child: OverflowBox(
                        maxWidth: double.infinity,
                        maxHeight: double.infinity,
                        child: SizedBox(
                          width: videoW * scale,
                          height: videoH * scale,
                          child: VideoPlayer(_controller),
                        ),
                      ),
                    );
                  },
                )
              : Container(color: const Color(0xFF0A3D3F)),
        ),

        // Shade — off by default (see overlayOpacity above).
        if (widget.overlayOpacity > 0)
          Positioned.fill(
            child: Container(
              color: Colors.black.withOpacity(widget.overlayOpacity),
            ),
          ),

        SafeArea(child: content),
      ],
    );
  }
}