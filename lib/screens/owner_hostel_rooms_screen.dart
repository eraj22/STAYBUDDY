import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';
import '../widgets/video_background.dart';
import 'owner_step_facilities.dart';

class OwnerHostelRoomsScreen extends StatefulWidget {
  final String ownerEmail, ownerName, hostelName, hostelType, city, address;
  final double latitude, longitude;

  const OwnerHostelRoomsScreen({
    super.key,
    required this.ownerEmail,
    required this.ownerName,
    required this.hostelName,
    required this.hostelType,
    required this.city,
    required this.address,
    required this.latitude,
    required this.longitude,
  });

  @override
  State<OwnerHostelRoomsScreen> createState() => _OwnerHostelRoomsScreenState();
}

class _OwnerHostelRoomsScreenState extends State<OwnerHostelRoomsScreen>
    with SingleTickerProviderStateMixin {
  final _floorsCtrl   = TextEditingController();
  final _capacityCtrl = TextEditingController();
  final _roomsCtrl    = TextEditingController();
  String? _error;

  // Room types with prices
  final _roomTypes = <Map<String, dynamic>>[
    {'type': 'Single',    'icon': Icons.bed_rounded,             'color': AppTheme.teal,              'selected': true,  'priceCtrl': TextEditingController(text: '15000')},
    {'type': 'Double',    'icon': Icons.king_bed_rounded,        'color': Color(0xFF185FA5),           'selected': true,  'priceCtrl': TextEditingController(text: '10000')},
    {'type': 'Triple',    'icon': Icons.bedroom_parent_rounded,  'color': Color(0xFF7B3FC4),           'selected': false, 'priceCtrl': TextEditingController(text: '8000')},
    {'type': 'Dormitory', 'icon': Icons.hotel_rounded,           'color': Color(0xFFB85C00),           'selected': false, 'priceCtrl': TextEditingController(text: '5000')},
  ];

  late final AnimationController _anim;
  late final Animation<double> _fade, _slide;

  @override
  void initState() {
    super.initState();
    _anim = AnimationController(vsync: this,
        duration: const Duration(milliseconds: 800));
    _fade  = CurvedAnimation(parent: _anim,
        curve: const Interval(0.15, 1.0, curve: Curves.easeOut));
    _slide = Tween<double>(begin: 60, end: 0).animate(CurvedAnimation(
        parent: _anim,
        curve: const Interval(0.15, 1.0, curve: Curves.easeOutCubic)));
    _anim.forward();
  }

  @override
  void dispose() {
    _anim.dispose();
    _floorsCtrl.dispose();
    _capacityCtrl.dispose();
    _roomsCtrl.dispose();
    for (final r in _roomTypes) {
      (r['priceCtrl'] as TextEditingController).dispose();
    }
    super.dispose();
  }

  void _next() {
    final floors   = int.tryParse(_floorsCtrl.text.trim());
    final capacity = int.tryParse(_capacityCtrl.text.trim());
    final rooms    = int.tryParse(_roomsCtrl.text.trim());

    if (floors == null || capacity == null || rooms == null) {
      setState(() => _error = 'Please enter valid numbers for all fields');
      return;
    }
    if (floors < 1 || capacity < 1 || rooms < 1) {
      setState(() => _error = 'All values must be at least 1');
      return;
    }
    setState(() => _error = null);

    Navigator.push(context, MaterialPageRoute(
      builder: (_) => OwnerStepFacilities(
        ownerEmail:  widget.ownerEmail,
        ownerName:   widget.ownerName,
        hostelName:  widget.hostelName,
        hostelType:  widget.hostelType,
        city:        widget.city,
        address:     widget.address,
        latitude:    widget.latitude,
        longitude:   widget.longitude,
        totalFloors: floors,
        totalRooms:  rooms,
        totalCapacity: capacity,
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
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(20, 14, 20, 0),
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Row(children: [
                    GestureDetector(
                      onTap: () => Navigator.pop(context),
                      child: Container(
                        width: 38, height: 38,
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.15),
                          shape: BoxShape.circle,
                          border: Border.all(color: Colors.white.withOpacity(0.3))),
                        child: const Icon(Icons.arrow_back_rounded,
                            color: Colors.white, size: 18),
                      ),
                    ),
                    const Spacer(),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 5),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.12),
                        borderRadius: BorderRadius.circular(20),
                        border: Border.all(color: Colors.white.withOpacity(0.2))),
                      child: Text('Step 4 of 9',
                          style: GoogleFonts.dmSans(
                              color: Colors.white, fontSize: 12,
                              fontWeight: FontWeight.w600)),
                    ),
                  ]),
                  const SizedBox(height: 16),
                  // Progress bar — 4 of 9
                  Row(children: List.generate(9, (i) => Expanded(
                    child: Container(
                      height: 4,
                      margin: EdgeInsets.only(right: i < 8 ? 4 : 0),
                      decoration: BoxDecoration(
                        color: i < 4
                            ? AppTheme.tealMint
                            : Colors.white.withOpacity(0.25),
                        borderRadius: BorderRadius.circular(2)),
                    ),
                  ))),
                  const SizedBox(height: 14),
                  Text('Room & Bed Details',
                      style: GoogleFonts.playfairDisplay(
                          fontSize: 30, fontWeight: FontWeight.w700,
                          color: Colors.white)),
                  const SizedBox(height: 4),
                  Text('Accommodation structure and pricing',
                      style: GoogleFonts.dmSans(
                          color: Colors.white.withOpacity(0.6), fontSize: 13)),
                ]),
              ),
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
                    maxChildSize: 0.96,
                    expand: true,
                    builder: (ctx, scrollCtrl) => _buildSheet(scrollCtrl),
                  ),
                ),
              ),
            ]),
          ),
        ),
      ),
    );
  }

  Widget _buildSheet(ScrollController scrollCtrl) {
    return Container(
      width: double.infinity,
      decoration: const BoxDecoration(
        color: AppTheme.bg,
        borderRadius: BorderRadius.vertical(top: Radius.circular(32)),
      ),
      child: Column(children: [
        const SizedBox(height: 12),
        Center(child: Container(
          width: 40, height: 4,
          decoration: BoxDecoration(
              color: AppTheme.border,
              borderRadius: BorderRadius.circular(2)))),
        Expanded(
          child: SingleChildScrollView(
            controller: scrollCtrl,
            padding: const EdgeInsets.fromLTRB(24, 14, 24, 32),
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              // Header
              Row(children: [
                Container(width: 42, height: 42,
                  decoration: BoxDecoration(
                      color: AppTheme.teal.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(12)),
                  child: const Icon(Icons.bed_rounded,
                      color: AppTheme.teal, size: 22)),
                const SizedBox(width: 12),
                Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text('Accommodation Details',
                      style: GoogleFonts.playfairDisplay(
                          fontSize: 19, fontWeight: FontWeight.w700,
                          color: AppTheme.textPrimary)),
                  Text('Step 4 of 9',
                      style: GoogleFonts.dmSans(
                          fontSize: 11, color: AppTheme.textMuted)),
                ]),
              ]),

              // Hostel info badge
              Container(
                margin: const EdgeInsets.symmetric(vertical: 14),
                padding: const EdgeInsets.all(13),
                decoration: BoxDecoration(
                  color: AppTheme.teal.withOpacity(0.05),
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(
                      color: AppTheme.teal.withOpacity(0.15), width: 0.5)),
                child: Row(children: [
                  const Icon(Icons.business_rounded,
                      size: 16, color: AppTheme.teal),
                  const SizedBox(width: 8),
                  Expanded(child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                    Text(widget.hostelName,
                        style: GoogleFonts.dmSans(
                            fontSize: 13, fontWeight: FontWeight.w700,
                            color: AppTheme.textPrimary)),
                    Text('${widget.city} · ${widget.hostelType}',
                        style: GoogleFonts.dmSans(
                            fontSize: 11, color: AppTheme.textMuted)),
                  ])),
                  Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                        color: const Color(0xFFEAF3DE),
                        borderRadius: BorderRadius.circular(8)),
                    child: Row(children: [
                      const Icon(Icons.location_on_rounded,
                          size: 11, color: Color(0xFF3B6D11)),
                      const SizedBox(width: 3),
                      Text('Pinned',
                          style: GoogleFonts.dmSans(
                              fontSize: 10, fontWeight: FontWeight.w600,
                              color: const Color(0xFF3B6D11))),
                    ]),
                  ),
                ]),
              ),

              // Counters
              _CounterField(label: 'Total Floors',
                  hint: 'Enter total floors',
                  icon: Icons.layers_rounded, ctrl: _floorsCtrl,
                  onIncrement: () {
                    final v = int.tryParse(_floorsCtrl.text) ?? 0;
                    setState(() => _floorsCtrl.text = (v + 1).toString());
                  },
                  onDecrement: () {
                    final v = int.tryParse(_floorsCtrl.text) ?? 0;
                    if (v > 1) setState(
                        () => _floorsCtrl.text = (v - 1).toString());
                  }),
              const SizedBox(height: 11),
              _CounterField(label: 'Total Capacity',
                  hint: 'Enter total capacity',
                  icon: Icons.people_rounded, ctrl: _capacityCtrl,
                  onIncrement: () {
                    final v = int.tryParse(_capacityCtrl.text) ?? 0;
                    setState(() => _capacityCtrl.text = (v + 1).toString());
                  },
                  onDecrement: () {
                    final v = int.tryParse(_capacityCtrl.text) ?? 0;
                    if (v > 1) setState(
                        () => _capacityCtrl.text = (v - 1).toString());
                  }),
              const SizedBox(height: 11),
              _CounterField(label: 'Total Rooms',
                  hint: 'Enter total rooms',
                  icon: Icons.meeting_room_rounded, ctrl: _roomsCtrl,
                  onIncrement: () {
                    final v = int.tryParse(_roomsCtrl.text) ?? 0;
                    setState(() => _roomsCtrl.text = (v + 1).toString());
                  },
                  onDecrement: () {
                    final v = int.tryParse(_roomsCtrl.text) ?? 0;
                    if (v > 1) setState(
                        () => _roomsCtrl.text = (v - 1).toString());
                  }),
              const SizedBox(height: 18),

              // Room types
              Row(children: [
                const Icon(Icons.category_rounded,
                    size: 14, color: AppTheme.teal),
                const SizedBox(width: 6),
                Text('Room Types & Pricing',
                    style: GoogleFonts.dmSans(
                        fontSize: 13, fontWeight: FontWeight.w700,
                        color: AppTheme.textPrimary)),
              ]),
              const SizedBox(height: 10),
              ...List.generate(_roomTypes.length, (i) {
                final r = _roomTypes[i];
                final sel = r['selected'] as bool;
                final color = r['color'] as Color;
                return Padding(
                  padding: const EdgeInsets.only(bottom: 8),
                  child: GestureDetector(
                    onTap: () => setState(
                        () => _roomTypes[i]['selected'] = !sel),
                    child: AnimatedContainer(
                      duration: const Duration(milliseconds: 180),
                      padding: const EdgeInsets.symmetric(
                          horizontal: 14, vertical: 11),
                      decoration: BoxDecoration(
                        color: sel
                            ? color.withOpacity(0.05)
                            : AppTheme.card,
                        borderRadius: BorderRadius.circular(14),
                        border: Border.all(
                            color: sel ? color : AppTheme.border,
                            width: sel ? 1.5 : 0.5)),
                      child: Row(children: [
                        Container(
                          width: 38, height: 38,
                          decoration: BoxDecoration(
                            color: sel
                                ? color.withOpacity(0.12)
                                : AppTheme.bgSecondary,
                            borderRadius: BorderRadius.circular(10)),
                          child: Icon(r['icon'] as IconData,
                              size: 18,
                              color: sel ? color : AppTheme.textMuted),
                        ),
                        const SizedBox(width: 12),
                        Expanded(child: Text(r['type'] as String,
                            style: GoogleFonts.dmSans(
                                fontSize: 13, fontWeight: FontWeight.w600,
                                color: sel
                                    ? color
                                    : AppTheme.textPrimary))),
                        if (sel) ...[
                          SizedBox(width: 100,
                            child: TextField(
                              controller: r['priceCtrl']
                                  as TextEditingController,
                              keyboardType: TextInputType.number,
                              textAlign: TextAlign.center,
                              style: GoogleFonts.dmSans(
                                  fontSize: 13,
                                  fontWeight: FontWeight.w700,
                                  color: color),
                              decoration: InputDecoration(
                                prefixText: 'PKR ',
                                prefixStyle: GoogleFonts.dmSans(
                                    fontSize: 10,
                                    color: AppTheme.textMuted),
                                contentPadding:
                                    const EdgeInsets.symmetric(
                                        horizontal: 8, vertical: 8),
                                isDense: true,
                                filled: true,
                                fillColor: color.withOpacity(0.06),
                                border: OutlineInputBorder(
                                  borderRadius:
                                      BorderRadius.circular(10),
                                  borderSide: BorderSide(
                                      color: color.withOpacity(0.3)),
                                ),
                                enabledBorder: OutlineInputBorder(
                                  borderRadius:
                                      BorderRadius.circular(10),
                                  borderSide: BorderSide(
                                      color: color.withOpacity(0.3),
                                      width: 0.5),
                                ),
                                focusedBorder: OutlineInputBorder(
                                  borderRadius:
                                      BorderRadius.circular(10),
                                  borderSide:
                                      BorderSide(color: color),
                                ),
                              ),
                            ),
                          ),
                        ] else ...[
                          Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 10, vertical: 5),
                            decoration: BoxDecoration(
                                color: AppTheme.bgSecondary,
                                borderRadius:
                                    BorderRadius.circular(8)),
                            child: Text('Off',
                                style: GoogleFonts.dmSans(
                                    fontSize: 11,
                                    color: AppTheme.textMuted)),
                          ),
                        ],
                      ]),
                    ),
                  ),
                );
              }),

              if (_error != null) ...[
                const SizedBox(height: 4),
                Container(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 14, vertical: 11),
                  decoration: BoxDecoration(
                    color: Colors.red.shade50,
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(
                        color: Colors.red.shade200, width: 0.5)),
                  child: Row(children: [
                    Icon(Icons.error_outline_rounded,
                        color: Colors.red.shade600, size: 16),
                    const SizedBox(width: 8),
                    Expanded(child: Text(_error!,
                        style: GoogleFonts.dmSans(
                            color: Colors.red.shade700,
                            fontSize: 12))),
                  ]),
                ),
              ],
              const SizedBox(height: 16),

              SizedBox(
                width: double.infinity, height: 52,
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
                    Text('Next  →  Facilities',
                        style: GoogleFonts.dmSans(
                            fontSize: 15,
                            fontWeight: FontWeight.w600)),
                    const SizedBox(width: 10),
                    Container(
                      width: 26, height: 26,
                      decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.2),
                          shape: BoxShape.circle),
                      child: const Icon(Icons.checklist_rounded,
                          size: 14),
                    ),
                  ]),
                ),
              ),
            ]),
          ),
        ),
      ]),
    );
  }
}

// ── Counter field widget (reused from original) ───────────────────────────────
class _CounterField extends StatelessWidget {
  final String label, hint;
  final IconData icon;
  final TextEditingController ctrl;
  final VoidCallback onIncrement, onDecrement;

  const _CounterField({
    required this.label, required this.hint, required this.icon,
    required this.ctrl, required this.onIncrement,
    required this.onDecrement,
  });

  @override
  Widget build(BuildContext context) {
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Row(children: [
        Icon(icon, size: 14, color: AppTheme.textMuted),
        const SizedBox(width: 6),
        Text(label, style: GoogleFonts.dmSans(
            fontSize: 12, fontWeight: FontWeight.w600,
            color: AppTheme.textMuted, letterSpacing: 0.3)),
      ]),
      const SizedBox(height: 6),
      Container(
        decoration: BoxDecoration(
          color: AppTheme.card,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: AppTheme.border, width: 0.5)),
        child: Row(children: [
          GestureDetector(
            onTap: onDecrement,
            child: Container(
              width: 46, height: 46,
              decoration: BoxDecoration(
                color: AppTheme.bg,
                borderRadius: const BorderRadius.horizontal(
                    left: Radius.circular(14)),
                border: Border(right: BorderSide(
                    color: AppTheme.border, width: 0.5))),
              child: const Icon(Icons.remove_rounded,
                  size: 18, color: AppTheme.textMuted),
            ),
          ),
          Expanded(
            child: TextField(
              controller: ctrl,
              textAlign: TextAlign.center,
              keyboardType: TextInputType.number,
              inputFormatters: [
                FilteringTextInputFormatter.digitsOnly,
                LengthLimitingTextInputFormatter(4),
              ],
              style: GoogleFonts.dmSans(
                  fontSize: 16, fontWeight: FontWeight.w700,
                  color: AppTheme.textPrimary),
              decoration: InputDecoration(
                hintText: hint,
                hintStyle: GoogleFonts.dmSans(
                    color: AppTheme.textMuted, fontSize: 13),
                border: InputBorder.none,
                contentPadding:
                    const EdgeInsets.symmetric(vertical: 12)),
            ),
          ),
          GestureDetector(
            onTap: onIncrement,
            child: Container(
              width: 46, height: 46,
              decoration: const BoxDecoration(
                color: AppTheme.teal,
                borderRadius: BorderRadius.horizontal(
                    right: Radius.circular(14))),
              child: const Icon(Icons.add_rounded,
                  size: 18, color: Colors.white),
            ),
          ),
        ]),
      ),
    ]);
  }
}