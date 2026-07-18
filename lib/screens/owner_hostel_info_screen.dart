import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';
import '../widgets/video_background.dart';
import 'owner_hostel_map_screen.dart';

class OwnerHostelInfoScreen extends StatefulWidget {
  final String ownerEmail, ownerName;
  const OwnerHostelInfoScreen({
    super.key,
    required this.ownerEmail,
    required this.ownerName,
  });

  @override
  State<OwnerHostelInfoScreen> createState() =>
      _OwnerHostelInfoScreenState();
}

class _OwnerHostelInfoScreenState extends State<OwnerHostelInfoScreen>
    with SingleTickerProviderStateMixin {
  final _nameCtrl    = TextEditingController();
  final _cityCtrl    = TextEditingController();
  final _addressCtrl = TextEditingController();
  String? _selectedType;
  String? _error;

  late final AnimationController _anim;
  late final Animation<double> _fade;
  late final Animation<double> _slide;

  // Only Girls and Boys
  final _types = [
    {
      'label': 'Girls Hostel',
      'subtitle': 'For female students only',
      'icon': Icons.female_rounded,
      'value': 'girls',
      'color': const Color(0xFFE91E8C),
    },
    {
      'label': 'Boys Hostel',
      'subtitle': 'For male students only',
      'icon': Icons.male_rounded,
      'value': 'boys',
      'color': const Color(0xFF185FA5),
    },
  ];

  @override
  void initState() {
    super.initState();
    _anim = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 800));
    _fade = CurvedAnimation(
        parent: _anim,
        curve: const Interval(0.15, 1.0, curve: Curves.easeOut));
    _slide = Tween<double>(begin: 60, end: 0).animate(CurvedAnimation(
        parent: _anim,
        curve: const Interval(0.15, 1.0, curve: Curves.easeOutCubic)));
    _anim.forward();
  }

  @override
  void dispose() {
    _anim.dispose();
    _nameCtrl.dispose();
    _cityCtrl.dispose();
    _addressCtrl.dispose();
    super.dispose();
  }

  void _next() {
    final name    = _nameCtrl.text.trim();
    final city    = _cityCtrl.text.trim();
    final address = _addressCtrl.text.trim();

    if (name.isEmpty || city.isEmpty ||
        address.isEmpty || _selectedType == null) {
      setState(() =>
          _error = 'Please fill all fields and select hostel type');
      return;
    }
    setState(() => _error = null);

    Navigator.push(context, MaterialPageRoute(
      builder: (_) => OwnerHostelMapScreen(
        ownerEmail:  widget.ownerEmail,
        ownerName:   widget.ownerName,
        hostelName:  name,
        hostelType:  _selectedType!,
        city:        city,
        address:     address,
      ),
    ));
  }

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        body: VideoBackground(
          assetPath: 'assets/images/background.mp4',
          overlayOpacity: 0.30,
          cardMargin: const EdgeInsets.all(14),
          cardRadius: 32,
          child: SafeArea(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // ── Top header ──────────────────────────────
                Padding(
                  padding: const EdgeInsets.fromLTRB(20, 14, 20, 0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      GestureDetector(
                        onTap: () => Navigator.pop(context),
                        child: Container(
                          width: 38, height: 38,
                          decoration: BoxDecoration(
                            color: Colors.white.withOpacity(0.15),
                            shape: BoxShape.circle,
                            border: Border.all(
                                color: Colors.white.withOpacity(0.3))),
                          child: const Icon(Icons.arrow_back_rounded,
                              color: Colors.white, size: 18)),
                      ),
                      const SizedBox(height: 16),
                      _StepIndicator(step: 3, total: 9),
                      const SizedBox(height: 14),
                      Text('Hostel Details',
                          style: GoogleFonts.playfairDisplay(
                              fontSize: 32,
                              fontWeight: FontWeight.w700,
                              color: Colors.white)),
                      const SizedBox(height: 4),
                      Text('Basic information about your hostel',
                          style: GoogleFonts.dmSans(
                              color: Colors.white.withOpacity(0.6),
                              fontSize: 13)),
                    ],
                  ),
                ),

                // ── Draggable sheet ─────────────────────────
                Expanded(
                  child: AnimatedBuilder(
                    animation: _anim,
                    builder: (_, child) => Transform.translate(
                      offset: Offset(0, _slide.value),
                      child: Opacity(opacity: _fade.value, child: child),
                    ),
                    child: DraggableScrollableSheet(
                      initialChildSize: 0.85,
                      minChildSize: 0.45,
                      maxChildSize: 0.95,
                      expand: true,
                      builder: (ctx, scrollCtrl) => Container(
                        decoration: const BoxDecoration(
                          color: AppTheme.bg,
                          borderRadius: BorderRadius.vertical(
                              top: Radius.circular(32)),
                        ),
                        child: Column(children: [
                          // Drag handle
                          Padding(
                            padding: const EdgeInsets.only(
                                top: 12, bottom: 6),
                            child: Center(
                              child: Container(
                                width: 40, height: 4,
                                decoration: BoxDecoration(
                                    color: AppTheme.border,
                                    borderRadius:
                                        BorderRadius.circular(2)),
                              ),
                            ),
                          ),
                          Expanded(
                            child: SingleChildScrollView(
                              controller: scrollCtrl,
                              padding: const EdgeInsets.fromLTRB(
                                  22, 8, 22, 32),
                              child: _buildContent(),
                            ),
                          ),
                        ]),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildContent() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Sheet header
        Row(children: [
          Container(
            width: 42, height: 42,
            decoration: BoxDecoration(
                color: AppTheme.teal.withOpacity(0.1),
                borderRadius: BorderRadius.circular(12)),
            child: const Icon(Icons.apartment_rounded,
                color: AppTheme.teal, size: 22)),
          const SizedBox(width: 12),
          Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text('Hostel Information',
                style: GoogleFonts.playfairDisplay(
                    fontSize: 20,
                    fontWeight: FontWeight.w700,
                    color: AppTheme.textPrimary)),
            Text('Step 3 of 9',
                style: GoogleFonts.dmSans(
                    fontSize: 11, color: AppTheme.textMuted)),
          ]),
        ]),
        const SizedBox(height: 20),

        // Hostel name
        _field(
            ctrl: _nameCtrl,
            hint: 'Enter hostel name',
            icon: Icons.business_rounded),
        const SizedBox(height: 16),

        // Hostel type — only Girls / Boys
        Text('Hostel Type',
            style: GoogleFonts.dmSans(
                fontSize: 13,
                fontWeight: FontWeight.w700,
                color: AppTheme.textPrimary)),
        const SizedBox(height: 10),
        Row(children: _types.map((t) {
          final sel    = _selectedType == t['value'];
          final color  = t['color'] as Color;
          return Expanded(
            child: Padding(
              padding: EdgeInsets.only(
                  right: t == _types.first ? 10 : 0),
              child: GestureDetector(
                onTap: () =>
                    setState(() => _selectedType = t['value'] as String),
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 200),
                  padding: const EdgeInsets.symmetric(
                      vertical: 18, horizontal: 16),
                  decoration: BoxDecoration(
                    color: sel ? color : AppTheme.card,
                    borderRadius: BorderRadius.circular(18),
                    border: Border.all(
                        color: sel ? color : AppTheme.border,
                        width: sel ? 2 : 0.5),
                    boxShadow: sel
                        ? [BoxShadow(
                            color: color.withOpacity(0.3),
                            blurRadius: 12,
                            offset: const Offset(0, 4))]
                        : [],
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        mainAxisAlignment:
                            MainAxisAlignment.spaceBetween,
                        children: [
                          Container(
                            width: 38, height: 38,
                            decoration: BoxDecoration(
                              color: sel
                                  ? Colors.white.withOpacity(0.2)
                                  : color.withOpacity(0.1),
                              shape: BoxShape.circle,
                            ),
                            child: Icon(
                              t['icon'] as IconData,
                              size: 20,
                              color: sel ? Colors.white : color,
                            ),
                          ),
                          if (sel)
                            Container(
                              width: 22, height: 22,
                              decoration: const BoxDecoration(
                                  color: Colors.white,
                                  shape: BoxShape.circle),
                              child: Icon(Icons.check_rounded,
                                  size: 14, color: color),
                            ),
                        ],
                      ),
                      const SizedBox(height: 12),
                      Text(t['label'] as String,
                          style: GoogleFonts.dmSans(
                              fontSize: 16,
                              fontWeight: FontWeight.w700,
                              color: sel
                                  ? Colors.white
                                  : AppTheme.textPrimary)),
                      const SizedBox(height: 3),
                      Text(t['subtitle'] as String,
                          style: GoogleFonts.dmSans(
                              fontSize: 11,
                              color: sel
                                  ? Colors.white.withOpacity(0.75)
                                  : AppTheme.textMuted)),
                    ],
                  ),
                ),
              ),
            ),
          );
        }).toList()),
        const SizedBox(height: 16),

        // City
        _field(
            ctrl: _cityCtrl,
            hint: 'City (e.g. Islamabad)',
            icon: Icons.location_city_rounded),
        const SizedBox(height: 11),

        // Address
        _field(
            ctrl: _addressCtrl,
            hint: 'Full address (street, sector, area)',
            icon: Icons.pin_drop_rounded,
            maxLines: 2),
        const SizedBox(height: 14),

        // Error
        if (_error != null) ...[
          Container(
            margin: const EdgeInsets.only(bottom: 12),
            padding: const EdgeInsets.symmetric(
                horizontal: 14, vertical: 11),
            decoration: BoxDecoration(
              color: Colors.red.shade50,
              borderRadius: BorderRadius.circular(12),
              border:
                  Border.all(color: Colors.red.shade200, width: 0.5)),
            child: Row(children: [
              Icon(Icons.error_outline_rounded,
                  color: Colors.red.shade600, size: 16),
              const SizedBox(width: 8),
              Expanded(
                  child: Text(_error!,
                      style: GoogleFonts.dmSans(
                          color: Colors.red.shade700, fontSize: 12))),
            ]),
          ),
        ],

        // Map tip
        Container(
          padding: const EdgeInsets.all(13),
          decoration: BoxDecoration(
            color: Colors.amber.shade50,
            borderRadius: BorderRadius.circular(14),
            border:
                Border.all(color: Colors.amber.shade200, width: 0.5)),
          child: Row(children: [
            Icon(Icons.location_on_rounded,
                color: Colors.amber.shade700, size: 18),
            const SizedBox(width: 10),
            Expanded(
                child: Text(
              "You'll pin your hostel location on the map in the next step.",
              style: GoogleFonts.dmSans(
                  color: Colors.amber.shade800, fontSize: 12),
            )),
          ]),
        ),
        const SizedBox(height: 20),

        // Next button
        SizedBox(
          width: double.infinity, height: 54,
          child: ElevatedButton(
            onPressed: _next,
            style: ElevatedButton.styleFrom(
              backgroundColor: AppTheme.teal,
              foregroundColor: Colors.white,
              elevation: 0,
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(16)),
            ),
            child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
              Text('Next: Pin on Map',
                  style: GoogleFonts.dmSans(
                      fontSize: 16, fontWeight: FontWeight.w600)),
              const SizedBox(width: 10),
              Container(
                width: 28, height: 28,
                decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.2),
                    shape: BoxShape.circle),
                child: const Icon(Icons.arrow_forward_rounded,
                    size: 16)),
            ]),
          ),
        ),
      ],
    );
  }

  Widget _field({
    required TextEditingController ctrl,
    required String hint,
    required IconData icon,
    int maxLines = 1,
  }) {
    return Container(
      decoration: BoxDecoration(
        color: AppTheme.card,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppTheme.border, width: 0.5),
      ),
      child: TextField(
        controller: ctrl,
        maxLines: maxLines,
        style: GoogleFonts.dmSans(
            fontSize: 14, color: AppTheme.textPrimary),
        decoration: InputDecoration(
          hintText: hint,
          hintStyle: GoogleFonts.dmSans(
              color: AppTheme.textMuted, fontSize: 13),
          prefixIcon:
              Icon(icon, size: 20, color: AppTheme.textMuted),
          border: InputBorder.none,
          contentPadding: const EdgeInsets.symmetric(
              horizontal: 16, vertical: 14),
        ),
      ),
    );
  }
}

// ── Step progress indicator ────────────────────────────────────────────────────
class _StepIndicator extends StatelessWidget {
  final int step, total;
  const _StepIndicator({required this.step, required this.total});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: List.generate(total, (i) => Expanded(
        child: Container(
          height: i == step - 1 ? 4 : 3,
          margin: EdgeInsets.only(right: i < total - 1 ? 5 : 0),
          decoration: BoxDecoration(
            color: i < step
                ? AppTheme.tealMint
                : Colors.white.withOpacity(0.25),
            borderRadius: BorderRadius.circular(2)),
        ),
      )),
    );
  }
}