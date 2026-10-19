import 'package:flutter/material.dart';
import 'responsive_shell.dart';
import 'video_background.dart';

/// Standard screen wrapper used across the whole app.
/// Wraps [child] in the looping background video + dark overlay + SafeArea,
/// inside a Scaffold, inside the ResponsiveShell (web/desktop letterboxing).
///
/// Using this everywhere means every screen gets the exact same
/// video-background treatment with one place to tune it, instead of each
/// screen re-nesting ResponsiveShell > Scaffold > VideoBackground by hand
/// (which is what caused the mismatched-parenthesis bugs earlier).
class VideoScaffold extends StatelessWidget {
  final Widget child;
  final String assetPath;
  final double overlayOpacity;
  final PreferredSizeWidget? appBar;
  final Widget? floatingActionButton;
  final Color? backgroundColor;
  final bool resizeToAvoidBottomInset;

  const VideoScaffold({
    super.key,
    required this.child,
    this.assetPath = 'assets/images/background.mp4',
    this.overlayOpacity = 0.0,
    this.appBar,
    this.floatingActionButton,
    this.backgroundColor,
    this.resizeToAvoidBottomInset = true,
  });

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        backgroundColor: backgroundColor ?? Colors.black,
        extendBodyBehindAppBar: appBar != null,
        appBar: appBar,
        resizeToAvoidBottomInset: resizeToAvoidBottomInset,
        floatingActionButton: floatingActionButton,
        body: VideoBackground(
          assetPath: assetPath,
          overlayOpacity: overlayOpacity,
          child: child,
        ),
      ),
    );
  }
}