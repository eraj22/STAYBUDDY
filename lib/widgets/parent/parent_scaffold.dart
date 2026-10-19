import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../theme.dart';
import '../responsive_shell.dart';
import '../video_background.dart';

class ParentScaffold extends StatefulWidget {
  final String title;

  final Widget child;

  final Widget? trailing;

  final bool showBackButton;

  final EdgeInsetsGeometry contentPadding;

  final bool scrollable;

  const ParentScaffold({
    super.key,
    required this.title,
    required this.child,
    this.trailing,
    this.showBackButton = true,
    this.contentPadding = const EdgeInsets.fromLTRB(
      18,
      10,
      18,
      30,
    ),
    this.scrollable = true,
  });

  @override
  State<ParentScaffold> createState() => _ParentScaffoldState();
}

class _ParentScaffoldState extends State<ParentScaffold> {

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        backgroundColor: Colors.transparent,
        body: VideoBackground(
          assetPath: 'assets/images/background.mp4',
          overlayOpacity: 0.46,
          cardMargin: const EdgeInsets.all(14),
          cardRadius: 32,
          child: SafeArea(
            child: Column(
              children: [
                _buildTopBar(),

                Expanded(
                  child: widget.scrollable
                      ? _buildScrollableContent()
                      : _buildNormalContent(),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildTopBar() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(
        10,
        8,
        16,
        8,
      ),
      child: Row(
        children: [
          if (widget.showBackButton)
            IconButton(
              onPressed: () {
                Navigator.pop(context);
              },
              icon: const Icon(
                Icons.arrow_back_ios_new_rounded,
                color: Colors.white,
                size: 20,
              ),
            )
          else
            const SizedBox(width: 8),

          if (widget.showBackButton)
            const SizedBox(width: 4),

          Expanded(
            child: Text(
              widget.title,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: GoogleFonts.dmSans(
                color: Colors.white,
                fontSize: 18,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),

          if (widget.trailing != null) ...[
            const SizedBox(width: 10),
            widget.trailing!,
          ] else
            Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: Colors.black.withOpacity(0.26),
                border: Border.all(
                  color: Colors.white.withOpacity(0.16),
                ),
              ),
              child: const Icon(
                Icons.family_restroom_rounded,
                color: AppTheme.tealMint,
                size: 19,
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildScrollableContent() {
    return LayoutBuilder(
      builder: (
        context,
        constraints,
      ) {
        return SingleChildScrollView(
          physics: const BouncingScrollPhysics(),
          keyboardDismissBehavior:
              ScrollViewKeyboardDismissBehavior.onDrag,
          padding: widget.contentPadding,
          child: Center(
            child: ConstrainedBox(
              constraints: const BoxConstraints(
                maxWidth: 600,
              ),
              child: widget.child,
            ),
          ),
        );
      },
    );
  }

  Widget _buildNormalContent() {
    return Padding(
      padding: widget.contentPadding,
      child: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(
            maxWidth: 600,
          ),
          child: widget.child,
        ),
      ),
    );
  }
}
