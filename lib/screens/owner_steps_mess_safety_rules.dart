// ═══ SHARED HELPER ════════════════════════════════════════════════════════════
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';
import '../widgets/video_background.dart';
import 'owner_step_images.dart';

// ─── Shared scaffold used by all remaining steps ───────────────────────────────
class _StepShell extends StatelessWidget {
  final int step;
  final String title, subtitle;
  final AnimationController anim;
  final Animation<double> fade, slide;
  final Widget sheet;

  const _StepShell({
    required this.step, required this.title, required this.subtitle,
    required this.anim, required this.fade, required this.slide,
    required this.sheet,
  });

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
            child: Column(crossAxisAlignment: CrossAxisAlignment.start,
                children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(20, 14, 20, 0),
                child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                  Row(children: [
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
                    const Spacer(),
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 12, vertical: 5),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.12),
                        borderRadius: BorderRadius.circular(20),
                        border: Border.all(
                            color: Colors.white.withOpacity(0.2))),
                      child: Text('Step $step of 9',
                          style: GoogleFonts.dmSans(
                              color: Colors.white, fontSize: 12,
                              fontWeight: FontWeight.w600)),
                    ),
                  ]),
                  const SizedBox(height: 16),
                  Row(children: List.generate(9, (i) => Expanded(
                    child: Container(
                      height: 4,
                      margin: EdgeInsets.only(right: i < 8 ? 4 : 0),
                      decoration: BoxDecoration(
                        color: i < step
                            ? AppTheme.tealMint
                            : Colors.white.withOpacity(0.25),
                        borderRadius: BorderRadius.circular(2)),
                    ),
                  ))),
                  const SizedBox(height: 14),
                  Text(title,
                      style: GoogleFonts.playfairDisplay(
                          fontSize: 30, fontWeight: FontWeight.w700,
                          color: Colors.white)),
                  const SizedBox(height: 4),
                  Text(subtitle,
                      style: GoogleFonts.dmSans(
                          color: Colors.white.withOpacity(0.6),
                          fontSize: 13)),
                ]),
              ),
              Expanded(
                child: AnimatedBuilder(
                  animation: anim,
                  builder: (_, child) => Transform.translate(
                    offset: Offset(0, slide.value),
                    child: Opacity(opacity: fade.value, child: child),
                  ),
                  child: DraggableScrollableSheet(
                    initialChildSize: 0.85,
                    minChildSize: 0.45,
                    maxChildSize: 0.96,
                    expand: true,
                    builder: (ctx, scrollCtrl) => _DraggableSheetWrapper(
                      scrollCtrl: scrollCtrl,
                      sheet: sheet,
                    ),
                  ),
                ),
              ),
            ]),
          ),
        ),
      ),
    );
  }
}

// ─── Shared sheet wrapper ──────────────────────────────────────────────────────
class _Sheet extends StatelessWidget {
  final List<Widget> children;
  const _Sheet({required this.children});

  @override
  Widget build(BuildContext context) {
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
            padding: const EdgeInsets.fromLTRB(20, 14, 20, 32),
            child: Column(crossAxisAlignment: CrossAxisAlignment.start,
                children: children),
          ),
        ),
      ]),
    );
  }
}

// ─── Shared widgets ────────────────────────────────────────────────────────────
Widget _label(String text, {IconData? icon}) => Padding(
  padding: const EdgeInsets.only(bottom: 8),
  child: Row(children: [
    if (icon != null) ...[Icon(icon, size: 14, color: AppTheme.teal),
      const SizedBox(width: 6)],
    Text(text, style: GoogleFonts.dmSans(
        fontSize: 13, fontWeight: FontWeight.w700,
        color: AppTheme.textPrimary)),
  ]),
);

Widget _yesNo(String label, bool? value, ValueChanged<bool> onChange) =>
    Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(children: [
        Expanded(child: Text(label, style: GoogleFonts.dmSans(
            fontSize: 13, color: AppTheme.textPrimary))),
        ...[true, false].map((v) {
          final sel = value == v;
          return Padding(
            padding: EdgeInsets.only(left: v ? 0 : 8),
            child: GestureDetector(
              onTap: () => onChange(v),
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 160),
                padding: const EdgeInsets.symmetric(
                    horizontal: 16, vertical: 7),
                decoration: BoxDecoration(
                  color: sel ? AppTheme.teal : AppTheme.card,
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(
                      color: sel ? AppTheme.teal : AppTheme.border,
                      width: 0.5)),
                child: Text(v ? 'Yes' : 'No',
                    style: GoogleFonts.dmSans(
                        fontSize: 12, fontWeight: FontWeight.w600,
                        color: sel ? Colors.white : AppTheme.textMuted)),
              ),
            ),
          );
        }),
      ]),
    );

Widget _check(String label, bool value, ValueChanged<bool> onChange,
    {IconData? icon}) =>
    GestureDetector(
      onTap: () => onChange(!value),
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 6),
        child: Row(children: [
          AnimatedContainer(
            duration: const Duration(milliseconds: 160),
            width: 22, height: 22,
            decoration: BoxDecoration(
              color: value ? AppTheme.teal : AppTheme.card,
              borderRadius: BorderRadius.circular(6),
              border: Border.all(
                  color: value ? AppTheme.teal : AppTheme.border,
                  width: 0.5)),
            child: value ? const Icon(Icons.check_rounded,
                color: Colors.white, size: 14) : null,
          ),
          const SizedBox(width: 10),
          if (icon != null) ...[Icon(icon, size: 15,
              color: AppTheme.textMuted), const SizedBox(width: 6)],
          Expanded(child: Text(label, style: GoogleFonts.dmSans(
              fontSize: 13, color: AppTheme.textPrimary))),
        ]),
      ),
    );

Widget _field(TextEditingController ctrl, String hint, IconData icon,
    {int maxLines = 1, TextInputType? type}) =>
    Container(
      decoration: BoxDecoration(
        color: AppTheme.card, borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppTheme.border, width: 0.5)),
      child: TextField(
        controller: ctrl, maxLines: maxLines, keyboardType: type,
        style: GoogleFonts.dmSans(
            fontSize: 14, color: AppTheme.textPrimary),
        decoration: InputDecoration(
          hintText: hint,
          hintStyle: GoogleFonts.dmSans(
              color: AppTheme.textMuted, fontSize: 13),
          prefixIcon: Icon(icon, size: 18, color: AppTheme.textMuted),
          border: InputBorder.none,
          contentPadding: const EdgeInsets.symmetric(
              horizontal: 16, vertical: 14)),
      ),
    );

Widget _nextBtn(BuildContext context, String label, IconData icon,
    Widget next) =>
    SizedBox(
      width: double.infinity, height: 52,
      child: ElevatedButton(
        onPressed: () => Navigator.push(
            context, MaterialPageRoute(builder: (_) => next)),
        style: ElevatedButton.styleFrom(
          backgroundColor: AppTheme.teal, foregroundColor: Colors.white,
          elevation: 0,
          shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16))),
        child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
          Text(label, style: GoogleFonts.dmSans(
              fontSize: 15, fontWeight: FontWeight.w600)),
          const SizedBox(width: 10),
          Container(
            width: 26, height: 26,
            decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.2),
                shape: BoxShape.circle),
            child: Icon(icon, size: 14)),
        ]),
      ),
    );

Widget _divider() =>
    Divider(height: 24, thickness: 0.5, color: AppTheme.border);

Widget _infoBox(String text) => Container(
  padding: const EdgeInsets.all(12),
  decoration: BoxDecoration(
    color: AppTheme.teal.withOpacity(0.05),
    borderRadius: BorderRadius.circular(12),
    border: Border(
        left: BorderSide(color: AppTheme.teal, width: 3))),
  child: Row(crossAxisAlignment: CrossAxisAlignment.start,
      children: [
    const Icon(Icons.info_outline_rounded,
        color: AppTheme.teal, size: 16),
    const SizedBox(width: 10),
    Expanded(child: Text(text, style: GoogleFonts.dmSans(
        fontSize: 12, color: AppTheme.teal, height: 1.4))),
  ]),
);


// ─── Draggable sheet wrapper for _Sheet ──────────────────────────────────────
class _DraggableSheetWrapper extends StatelessWidget {
  final ScrollController scrollCtrl;
  final Widget sheet;
  const _DraggableSheetWrapper({required this.scrollCtrl, required this.sheet});

  @override
  Widget build(BuildContext context) => sheet;
}


// ════════════════════════════════════════════════════════════════════════════
// STEP 6 — Mess & Food
// ════════════════════════════════════════════════════════════════════════════
class OwnerStepMess extends StatefulWidget {
  final String ownerEmail, ownerName, hostelName, hostelType, city, address;
  final double latitude, longitude;

  const OwnerStepMess({
    super.key,
    required this.ownerEmail, required this.ownerName,
    required this.hostelName, required this.hostelType,
    required this.city, required this.address,
    required this.latitude, required this.longitude,
  });

  @override
  State<OwnerStepMess> createState() => _OwnerStepMessState();
}

class _OwnerStepMessState extends State<OwnerStepMess>
    with SingleTickerProviderStateMixin {
  bool _messIncluded = true;
  bool _extraCharges = false;
  final _chargesCtrl = TextEditingController();
  int _mealsPerDay = 3;
  String _mealType = 'Both';
  final _menuCtrl = TextEditingController();
  final _menu = <String>[];
  final _timings = <Map<String, String>>[];
  final _timingCtrl = TextEditingController();

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
    _chargesCtrl.dispose(); _menuCtrl.dispose(); _timingCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return _StepShell(
      step: 6, title: 'Mess & Food Details',
      subtitle: 'Food arrangements for residents',
      anim: _anim, fade: _fade, slide: _slide,
      sheet: _Sheet(children: [
        _yesNo('Is mess included in rent?', _messIncluded,
            (v) => setState(() => _messIncluded = v)),

        if (!_messIncluded) ...[
          _yesNo('Additional mess charges available?', _extraCharges,
              (v) => setState(() => _extraCharges = v)),
          if (_extraCharges) ...[
            _field(_chargesCtrl, 'Monthly mess charges (PKR)',
                Icons.payments_outlined, type: TextInputType.number),
            const SizedBox(height: 10),
          ],
        ],

        _divider(),
        _label('Meals per day', icon: Icons.restaurant_rounded),
        Row(mainAxisAlignment: MainAxisAlignment.center, children: [
          _cBtn(Icons.remove_rounded, () =>
              setState(() { if (_mealsPerDay > 1) _mealsPerDay--; })),
          Padding(padding: const EdgeInsets.symmetric(horizontal: 24),
            child: Text('$_mealsPerDay', style: GoogleFonts.dmSans(
                fontSize: 24, fontWeight: FontWeight.w700,
                color: AppTheme.teal))),
          _cBtn(Icons.add_rounded, () =>
              setState(() { if (_mealsPerDay < 5) _mealsPerDay++; })),
        ]),
        const SizedBox(height: 12),

        _label('Meal Type', icon: Icons.set_meal_rounded),
        Wrap(spacing: 8, runSpacing: 8,
          children: ['Veg Only', 'Non-Veg Only', 'Both'].map((t) =>
            GestureDetector(
              onTap: () => setState(() => _mealType = t),
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 160),
                padding: const EdgeInsets.symmetric(
                    horizontal: 14, vertical: 8),
                decoration: BoxDecoration(
                  color: _mealType == t ? AppTheme.teal : AppTheme.card,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(
                      color: _mealType == t
                          ? AppTheme.teal : AppTheme.border,
                      width: 0.5)),
                child: Text(t, style: GoogleFonts.dmSans(
                    fontSize: 12, fontWeight: FontWeight.w600,
                    color: _mealType == t
                        ? Colors.white : AppTheme.textPrimary)),
              ),
            )).toList(),
        ),
        const SizedBox(height: 14),

        _label('Meal Timings', icon: Icons.schedule_rounded),
        ..._timings.map((t) => Container(
          margin: const EdgeInsets.only(bottom: 6),
          padding: const EdgeInsets.symmetric(
              horizontal: 14, vertical: 10),
          decoration: BoxDecoration(
            color: AppTheme.card, borderRadius: BorderRadius.circular(12),
            border: Border.all(color: AppTheme.border, width: 0.5)),
          child: Row(children: [
            const Icon(Icons.schedule_rounded,
                size: 15, color: AppTheme.teal),
            const SizedBox(width: 10),
            Expanded(child: Text('${t['label']}: ${t['time']}',
                style: GoogleFonts.dmSans(
                    fontSize: 12, color: AppTheme.textPrimary))),
          ]),
        )),
        Row(children: [
          Expanded(child: _field(_timingCtrl,
              'Add timing (e.g. Tea: 5 PM)',
              Icons.access_time_rounded)),
          const SizedBox(width: 8),
          _addBtn(() {
            final v = _timingCtrl.text.trim();
            if (v.isNotEmpty) setState(() {
              _timings.add({'label': 'Other', 'time': v});
              _timingCtrl.clear();
            });
          }),
        ]),
        const SizedBox(height: 14),

        _label('Menu Items (Optional)', icon: Icons.menu_book_rounded),
        if (_menu.isNotEmpty) ...[
          Wrap(spacing: 8, runSpacing: 8,
            children: _menu.map((m) => Chip(
              label: Text(m, style: GoogleFonts.dmSans(fontSize: 12)),
              backgroundColor: AppTheme.green.withOpacity(0.08),
              side: BorderSide(
                  color: AppTheme.green.withOpacity(0.3)),
              deleteIconColor: AppTheme.green,
              onDeleted: () => setState(() => _menu.remove(m)),
            )).toList(),
          ),
          const SizedBox(height: 8),
        ],
        Row(children: [
          Expanded(child: _field(_menuCtrl,
              'e.g. Daal Chawal, Biryani, Roti',
              Icons.restaurant_menu_rounded)),
          const SizedBox(width: 8),
          _addBtn(() {
            final v = _menuCtrl.text.trim();
            if (v.isNotEmpty) setState(() {
              _menu.add(v); _menuCtrl.clear();
            });
          }),
        ]),
        const SizedBox(height: 20),

        _nextBtn(context, 'Next  →  Safety & Security',
            Icons.security_rounded,
            OwnerStepSafety(
              ownerEmail: widget.ownerEmail,
              ownerName: widget.ownerName,
              hostelName: widget.hostelName,
              hostelType: widget.hostelType,
              city: widget.city, address: widget.address,
              latitude: widget.latitude, longitude: widget.longitude,
            )),
      ]),
    );
  }

  Widget _cBtn(IconData icon, VoidCallback onTap) => GestureDetector(
    onTap: onTap,
    child: Container(
      width: 40, height: 40,
      decoration: BoxDecoration(
          color: AppTheme.teal.withOpacity(0.1), shape: BoxShape.circle),
      child: Icon(icon, size: 18, color: AppTheme.teal)),
  );

  Widget _addBtn(VoidCallback onTap) => GestureDetector(
    onTap: onTap,
    child: Container(
      width: 48, height: 48,
      decoration: BoxDecoration(
          color: AppTheme.teal,
          borderRadius: BorderRadius.circular(14)),
      child: const Icon(Icons.add_rounded, color: Colors.white, size: 22)),
  );
}


// ════════════════════════════════════════════════════════════════════════════
// STEP 7 — Safety & Security
// ════════════════════════════════════════════════════════════════════════════
class OwnerStepSafety extends StatefulWidget {
  final String ownerEmail, ownerName, hostelName, hostelType, city, address;
  final double latitude, longitude;

  const OwnerStepSafety({
    super.key,
    required this.ownerEmail, required this.ownerName,
    required this.hostelName, required this.hostelType,
    required this.city, required this.address,
    required this.latitude, required this.longitude,
  });

  @override
  State<OwnerStepSafety> createState() => _OwnerStepSafetyState();
}

class _OwnerStepSafetyState extends State<OwnerStepSafety>
    with SingleTickerProviderStateMixin {
  bool _attendance = true;
  bool _entryCard = false;
  bool _cctv = true;
  final _cctvAreas = {
    'Entrance': true, 'Rooms': false,
    'Corridors': true, 'Common Areas': true, 'Parking': false,
  };
  bool _warden = true;
  final _guardShifts = {'Day': false, 'Night': true, '24 Hours': false};
  bool _hasCurfew = true;
  TimeOfDay _curfew = const TimeOfDay(hour: 22, minute: 0);

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
  void dispose() { _anim.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    return _StepShell(
      step: 7, title: 'Safety & Security',
      subtitle: 'How do you keep residents safe?',
      anim: _anim, fade: _fade, slide: _slide,
      sheet: _Sheet(children: [
        _infoBox('Parents pay close attention to safety — a detailed setup builds trust.'),
        const SizedBox(height: 14),

        _yesNo('Attendance system available?', _attendance,
            (v) => setState(() => _attendance = v)),
        _yesNo('Entry card / access system?', _entryCard,
            (v) => setState(() => _entryCard = v)),
        _yesNo('CCTV cameras installed?', _cctv,
            (v) => setState(() => _cctv = v)),

        if (_cctv) ...[
          _label('CCTV Coverage', icon: Icons.videocam_rounded),
          ..._cctvAreas.entries.map((e) =>
              _check(e.key, e.value,
                  (v) => setState(() => _cctvAreas[e.key] = v))),
          const SizedBox(height: 8),
        ],

        _divider(),
        _yesNo('Is warden onsite?', _warden,
            (v) => setState(() => _warden = v)),

        _label('Security Guard Shifts', icon: Icons.security_rounded),
        ..._guardShifts.entries.map((e) =>
            _check(e.key, e.value, (v) => setState(() {
              _guardShifts[e.key] = v;
              if (e.key == '24 Hours' && v) {
                _guardShifts['Day'] = false;
                _guardShifts['Night'] = false;
              }
            }))),
        const SizedBox(height: 10),

        _divider(),
        _yesNo('Is there a curfew time?', _hasCurfew,
            (v) => setState(() => _hasCurfew = v)),

        if (_hasCurfew) ...[
          _label('Curfew Time', icon: Icons.lock_clock_rounded),
          GestureDetector(
            onTap: () async {
              final t = await showTimePicker(
                context: context, initialTime: _curfew,
                builder: (ctx, child) => Theme(
                  data: Theme.of(ctx).copyWith(
                    colorScheme: const ColorScheme.light(
                        primary: AppTheme.teal,
                        onPrimary: Colors.white)),
                  child: child!),
              );
              if (t != null) setState(() => _curfew = t);
            },
            child: Container(
              padding: const EdgeInsets.symmetric(
                  horizontal: 16, vertical: 14),
              decoration: BoxDecoration(
                color: AppTheme.card,
                borderRadius: BorderRadius.circular(14),
                border: Border.all(
                    color: AppTheme.teal.withOpacity(0.4))),
              child: Row(children: [
                const Icon(Icons.access_time_rounded,
                    color: AppTheme.teal, size: 20),
                const SizedBox(width: 12),
                Text(_curfew.format(context),
                    style: GoogleFonts.dmSans(
                        fontSize: 16, fontWeight: FontWeight.w700,
                        color: AppTheme.teal)),
                const Spacer(),
                const Icon(Icons.edit_rounded,
                    size: 16, color: AppTheme.textMuted),
              ]),
            ),
          ),
        ],
        const SizedBox(height: 20),

        _nextBtn(context, 'Next  →  Rules & Policies',
            Icons.rule_rounded,
            OwnerStepRules(
              ownerEmail: widget.ownerEmail,
              ownerName: widget.ownerName,
              hostelName: widget.hostelName,
              hostelType: widget.hostelType,
              city: widget.city, address: widget.address,
              latitude: widget.latitude, longitude: widget.longitude,
            )),
      ]),
    );
  }
}


// ════════════════════════════════════════════════════════════════════════════
// STEP 8 — Rules & Policies
// ════════════════════════════════════════════════════════════════════════════
class OwnerStepRules extends StatefulWidget {
  final String ownerEmail, ownerName, hostelName, hostelType, city, address;
  final double latitude, longitude;

  const OwnerStepRules({
    super.key,
    required this.ownerEmail, required this.ownerName,
    required this.hostelName, required this.hostelType,
    required this.city, required this.address,
    required this.latitude, required this.longitude,
  });

  @override
  State<OwnerStepRules> createState() => _OwnerStepRulesState();
}

class _OwnerStepRulesState extends State<OwnerStepRules>
    with SingleTickerProviderStateMixin {
  final _attendanceTypes = {
    'Register-Based': true, 'Digital': false, 'Biometric': false};
  bool _lateEntry = false;
  bool _parentNotif = true;
  final _guestPolicy = {
    'Allowed': false, 'Not Allowed': true, 'With Permission': false};
  final _paymentMethods = {
    'Cash': true, 'Bank Transfer': false,
    'JazzCash': false, 'EasyPaisa': false};
  bool _finePolicy = false;
  final _fineCtrl = TextEditingController();
  final _rules = <String>[];
  final _ruleCtrl = TextEditingController();

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
    _anim.dispose(); _fineCtrl.dispose(); _ruleCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return _StepShell(
      step: 8, title: 'Rules & Policies',
      subtitle: 'Hostel regulations and payment',
      anim: _anim, fade: _fade, slide: _slide,
      sheet: _Sheet(children: [
        _label('Attendance Type', icon: Icons.how_to_reg_rounded),
        ..._attendanceTypes.entries.map((e) =>
            _check(e.key, e.value,
                (v) => setState(() => _attendanceTypes[e.key] = v))),
        _divider(),

        _yesNo('Late entry allowed?', _lateEntry,
            (v) => setState(() => _lateEntry = v)),
        _yesNo('Parent notified on absence?', _parentNotif,
            (v) => setState(() => _parentNotif = v)),
        _divider(),

        _label('Guest Policy', icon: Icons.people_rounded),
        ..._guestPolicy.entries.map((e) =>
            _check(e.key, e.value, (v) => setState(() {
              for (final k in _guestPolicy.keys) _guestPolicy[k] = false;
              _guestPolicy[e.key] = v;
            }))),
        _divider(),

        _label('Payment Methods', icon: Icons.payments_rounded),
        Wrap(spacing: 8, runSpacing: 8,
          children: _paymentMethods.entries.map((e) => GestureDetector(
            onTap: () =>
                setState(() => _paymentMethods[e.key] = !e.value),
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 160),
              padding: const EdgeInsets.symmetric(
                  horizontal: 14, vertical: 8),
              decoration: BoxDecoration(
                color: e.value ? AppTheme.teal : AppTheme.card,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(
                    color: e.value ? AppTheme.teal : AppTheme.border,
                    width: 0.5)),
              child: Text(e.key, style: GoogleFonts.dmSans(
                  fontSize: 12, fontWeight: FontWeight.w600,
                  color: e.value
                      ? Colors.white : AppTheme.textPrimary)),
            ),
          )).toList(),
        ),
        _divider(),

        _yesNo('Fine policy for violations?', _finePolicy,
            (v) => setState(() => _finePolicy = v)),
        if (_finePolicy) ...[
          _field(_fineCtrl,
              'Describe fine policy (e.g. PKR 500/late entry)',
              Icons.gavel_rounded, maxLines: 2),
          const SizedBox(height: 10),
        ],

        _label('Additional Rules', icon: Icons.rule_rounded),
        ..._rules.map((r) => Container(
          margin: const EdgeInsets.only(bottom: 6),
          padding: const EdgeInsets.symmetric(
              horizontal: 12, vertical: 10),
          decoration: BoxDecoration(
            color: AppTheme.card,
            borderRadius: BorderRadius.circular(10),
            border: Border.all(color: AppTheme.border, width: 0.5)),
          child: Row(children: [
            Container(width: 6, height: 6,
              decoration: const BoxDecoration(
                  color: AppTheme.teal, shape: BoxShape.circle)),
            const SizedBox(width: 10),
            Expanded(child: Text(r, style: GoogleFonts.dmSans(
                fontSize: 13, color: AppTheme.textPrimary))),
            GestureDetector(
              onTap: () => setState(() => _rules.remove(r)),
              child: const Icon(Icons.close_rounded,
                  size: 16, color: AppTheme.textMuted)),
          ]),
        )),
        Row(children: [
          Expanded(child: _field(_ruleCtrl,
              'e.g. No smoking on premises',
              Icons.add_rounded)),
          const SizedBox(width: 8),
          GestureDetector(
            onTap: () {
              final v = _ruleCtrl.text.trim();
              if (v.isNotEmpty) setState(() {
                _rules.add(v); _ruleCtrl.clear();
              });
            },
            child: Container(
              width: 48, height: 48,
              decoration: BoxDecoration(
                  color: AppTheme.teal,
                  borderRadius: BorderRadius.circular(14)),
              child: const Icon(Icons.add_rounded,
                  color: Colors.white, size: 22)),
          ),
        ]),
        const SizedBox(height: 20),

        _nextBtn(context, 'Next  →  Photos & Videos',
            Icons.photo_library_rounded,
            OwnerStepImages(
              ownerEmail: widget.ownerEmail,
              ownerName: widget.ownerName,
              hostelName: widget.hostelName,
              hostelType: widget.hostelType,
              city: widget.city, address: widget.address,
              latitude: widget.latitude, longitude: widget.longitude,
            )),
      ]),
    );
  }
}