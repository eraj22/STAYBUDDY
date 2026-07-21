import 'package:flutter/material.dart';

class ThreeDContainer extends StatefulWidget {
  final Widget child;
  final double maxTilt;
  final bool enableGloss;
  final double borderRadius;

  const ThreeDContainer({
    super.key,
    required this.child,
    this.maxTilt = 12.0, // max tilt in degrees
    this.enableGloss = true,
    this.borderRadius = 22.0,
  });

  @override
  State<ThreeDContainer> createState() => _ThreeDContainerState();
}

class _ThreeDContainerState extends State<ThreeDContainer>
    with SingleTickerProviderStateMixin {
  double _tiltX = 0.0;
  double _tiltY = 0.0;
  bool _hovering = false;
  late final AnimationController _animController;
  late Animation<double> _tiltXAnim;
  late Animation<double> _tiltYAnim;

  @override
  void initState() {
    super.initState();
    _animController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 250),
    );
  }

  @override
  void dispose() {
    _animController.dispose();
    super.dispose();
  }

  void _updateTilt(double dx, double dy, double width, double height) {
    if (_animController.isAnimating) return;
    // Normalize position relative to center of the widget: range [-1, 1]
    final normX = (dx - width / 2) / (width / 2);
    final normY = (dy - height / 2) / (height / 2);

    setState(() {
      // Rotate around X axis for vertical movement (normalized Y)
      // Rotate around Y axis for horizontal movement (normalized X)
      _tiltX = -normY * widget.maxTilt * (3.14159265 / 180);
      _tiltY = normX * widget.maxTilt * (3.14159265 / 180);
    });
  }

  void _resetTilt() {
    _tiltXAnim = Tween<double>(begin: _tiltX, end: 0.0).animate(
      CurvedAnimation(parent: _animController, curve: Curves.easeOut),
    );
    _tiltYAnim = Tween<double>(begin: _tiltY, end: 0.0).animate(
      CurvedAnimation(parent: _animController, curve: Curves.easeOut),
    );
    _animController.forward(from: 0.0).then((_) {
      setState(() {
        _tiltX = 0.0;
        _tiltY = 0.0;
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    return MouseRegion(
      onEnter: (_) => setState(() => _hovering = true),
      onExit: (_) {
        setState(() => _hovering = false);
        _resetTilt();
      },
      onHover: (event) {
        final box = context.findRenderObject() as RenderBox?;
        if (box != null) {
          final local = box.globalToLocal(event.position);
          _updateTilt(local.dx, local.dy, box.size.width, box.size.height);
        }
      },
      child: GestureDetector(
        onPanUpdate: (details) {
          final box = context.findRenderObject() as RenderBox?;
          if (box != null) {
            final local = box.globalToLocal(details.globalPosition);
            _updateTilt(local.dx, local.dy, box.size.width, box.size.height);
          }
        },
        onPanEnd: (_) => _resetTilt(),
        child: AnimatedBuilder(
          animation: _animController,
          builder: (context, child) {
            final tx = _animController.isAnimating ? _tiltXAnim.value : _tiltX;
            final ty = _animController.isAnimating ? _tiltYAnim.value : _tiltY;
            return Transform(
              transform: Matrix4.identity()
                ..setEntry(3, 2, 0.0015) // perspective depth
                ..rotateX(tx)
                ..rotateY(ty),
              alignment: FractionalOffset.center,
              child: Stack(
                children: [
                  widget.child,
                  if (widget.enableGloss && (_hovering || _tiltX != 0 || _tiltY != 0))
                    Positioned.fill(
                      child: IgnorePointer(
                        child: Container(
                          decoration: BoxDecoration(
                            borderRadius: BorderRadius.circular(widget.borderRadius),
                            gradient: LinearGradient(
                              begin: Alignment(ty * 2 - 1.0, -tx * 2 - 1.0),
                              end: Alignment(-ty * 2 + 1.0, tx * 2 + 1.0),
                              colors: [
                                Colors.white.withOpacity(0.18),
                                Colors.white.withOpacity(0.0),
                                Colors.black.withOpacity(0.08),
                              ],
                            ),
                          ),
                        ),
                      ),
                    ),
                ],
              ),
            );
          },
        ),
      ),
    );
  }
}
